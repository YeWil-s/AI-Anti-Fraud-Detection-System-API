import api from '../js/api.js';

export default {
    template: `
        <div>
            <el-row :gutter="20" class="stat-row" v-loading="loading">
                <el-col :span="6">
                    <div class="stat-card">
                        <div class="stat-icon" style="background:#e0e7ff; color:#4f46e5"><i class="ri-user-smile-line"></i></div>
                        <div class="stat-info">
                            <div class="num">{{ stats.total_users || 0 }}</div>
                            <div class="label">注册用户总数</div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card">
                        <div class="stat-icon" style="background:#d1fae5; color:#059669"><i class="ri-phone-line"></i></div>
                        <div class="stat-info">
                            <div class="num">{{ stats.total_calls || 0 }}</div>
                            <div class="label">累计通话检测</div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card">
                        <div class="stat-icon" style="background:#fee2e2; color:#dc2626"><i class="ri-shield-cross-line"></i></div>
                        <div class="stat-info">
                            <div class="num">{{ stats.fraud_blocked || 0 }}</div>
                            <div class="label">拦截诈骗次数</div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card">
                        <div class="stat-icon" style="background:#fef3c7; color:#d97706"><i class="ri-database-2-line"></i></div>
                        <div class="stat-info">
                            <div class="num">{{ stats.active_rules || 0 }}</div>
                            <div class="label">生效风控规则</div>
                        </div>
                    </div>
                </el-col>
            </el-row>

            <el-row :gutter="20">
                <el-col :span="16">
                    <div class="page-card" style="height: 400px;">
                        <div class="page-title">系统防御趋势 (模拟展示)</div>
                        <div id="chartLine" style="width:100%; height:320px; margin-top:20px;"></div>
                    </div>
                </el-col>
                <el-col :span="8">
                    <div class="page-card" style="height: 400px;">
                        <div class="page-title">风控构成</div>
                        <div id="chartPie" style="width:100%; height:320px; margin-top:20px;"></div>
                    </div>
                </el-col>
            </el-row>
        </div>
    `,
    data() {
        return {
            loading: false,
            stats: {}
        }
    },
    async mounted() {
        await this.loadStats();
        this.initCharts();
    },
    methods: {
        async loadStats() {
            this.loading = true;
            try {
                this.stats = await api.getStats();
            } catch(e) { console.error(e) } 
            finally { this.loading = false; }
        },
        initCharts() {
            // 折线图
            const chart1 = echarts.init(document.getElementById('chartLine'));
            chart1.setOption({
                grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
                xAxis: { type: 'category', data: ['周一','周二','周三','周四','周五','周六','周日'] },
                yAxis: { type: 'value' },
                series: [{ data: [12, 19, 15, 28, 35, 22, 40], type: 'line', smooth: true, itemStyle:{color:'#4f46e5'}, areaStyle:{color:'#e0e7ff'} }]
            });
            // 饼图
            const chart2 = echarts.init(document.getElementById('chartPie'));
            chart2.setOption({
                tooltip: { trigger: 'item' },
                series: [{
                    type: 'pie', radius: ['40%', '70%'],
                    data: [
                        { value: this.stats.fraud_blocked || 10, name: '诈骗拦截' },
                        { value: this.stats.total_calls || 100, name: '正常通话' }
                    ]
                }]
            });
        }
    }
}