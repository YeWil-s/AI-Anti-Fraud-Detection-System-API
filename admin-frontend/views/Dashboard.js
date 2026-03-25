import api from '../js/api.js';

export default {
    template: `
        <div>
            <!-- 核心指标卡片 -->
            <el-row :gutter="20" class="stat-row" v-loading="loading">
                <el-col :span="6">
                    <div class="stat-card">
                        <div class="stat-icon" style="background:#e0e7ff; color:#4f46e5"><i class="ri-user-smile-line"></i></div>
                        <div class="stat-info">
                            <div class="num">{{ stats.total_users || 0 }}</div>
                            <div class="label">注册用户总数</div>
                            <div class="trend">
                                <el-tag size="small" type="success">+{{ stats.new_users_today || 0 }} 今日</el-tag>
                                <span style="margin-left: 8px; font-size: 12px;" :style="{color: (stats.users_growth_rate || 0) >= 0 ? '#10b981' : '#dc2626'}">
                                    {{ formatGrowthRate(stats.users_growth_rate) }}
                                </span>
                            </div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card">
                        <div class="stat-icon" style="background:#d1fae5; color:#059669"><i class="ri-phone-line"></i></div>
                        <div class="stat-info">
                            <div class="num">{{ stats.total_calls || 0 }}</div>
                            <div class="label">累计通话检测</div>
                            <div class="trend">
                                <el-tag size="small" type="info">{{ stats.detections_today || 0 }} 今日</el-tag>
                                <span style="margin-left: 8px; font-size: 12px;" :style="{color: (stats.calls_growth_rate || 0) >= 0 ? '#10b981' : '#dc2626'}">
                                    {{ formatGrowthRate(stats.calls_growth_rate) }}
                                </span>
                            </div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card">
                        <div class="stat-icon" style="background:#fee2e2; color:#dc2626"><i class="ri-shield-cross-line"></i></div>
                        <div class="stat-info">
                            <div class="num">{{ stats.fraud_blocked || 0 }}</div>
                            <div class="label">拦截诈骗次数</div>
                            <div class="trend">
                                <el-tag size="small" type="danger">{{ stats.blocked_today || 0 }} 今日</el-tag>
                                <span style="margin-left: 8px; font-size: 12px;" :style="{color: (stats.blocked_growth_rate || 0) >= 0 ? '#10b981' : '#dc2626'}">
                                    {{ formatGrowthRate(stats.blocked_growth_rate) }}
                                </span>
                            </div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card">
                        <div class="stat-icon" style="background:#fef3c7; color:#d97706"><i class="ri-database-2-line"></i></div>
                        <div class="stat-info">
                            <div class="num">{{ stats.active_rules || 0 }}</div>
                            <div class="label">生效风控规则</div>
                            <div class="trend">
                                <el-tag size="small" type="warning">{{ stats.blacklist_count || 0 }} 黑名单</el-tag>
                            </div>
                        </div>
                    </div>
                </el-col>
            </el-row>

            <!-- 第二行指标 -->
            <el-row :gutter="20" style="margin-top: 20px;">
                <el-col :span="6">
                    <div class="stat-card stat-card-gradient-purple">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ stats.detection_rate || 0 }}%</div>
                            <div class="label">诈骗检出率</div>
                            <div style="font-size: 12px; margin-top: 5px; opacity: 0.8;">
                                平均风险分: {{ stats.avg_risk_score || 0 }}
                            </div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card stat-card-gradient-pink">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ systemHealth.pending_cases || 0 }}</div>
                            <div class="label">待学习案例</div>
                            <div style="font-size: 12px; margin-top: 5px; opacity: 0.8;">
                                已学习: {{ systemHealth.learned_cases || 0 }}
                            </div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card stat-card-gradient-blue">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ recentDetections.length || 0 }}</div>
                            <div class="label">24H检测次数</div>
                            <div style="font-size: 12px; margin-top: 5px; opacity: 0.8;">
                                实时监控中
                            </div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card stat-card-gradient-green">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ stats.system_health || '100%' }}</div>
                            <div class="label">系统健康度</div>
                            <div style="font-size: 12px; margin-top: 5px; opacity: 0.8;">
                                运行正常
                            </div>
                        </div>
                    </div>
                </el-col>
            </el-row>

            <!-- 图表区域 -->
            <el-row :gutter="20" style="margin-top: 20px;">
                <el-col :span="16">
                    <div class="page-card">
                        <div class="page-header">
                            <div class="page-title">检测趋势分析</div>
                            <el-radio-group v-model="trendDays" size="small" @change="loadTrendData">
                                <el-radio-button :label="7">近7天</el-radio-button>
                                <el-radio-button :label="14">近14天</el-radio-button>
                                <el-radio-button :label="30">近30天</el-radio-button>
                            </el-radio-group>
                        </div>
                        <div v-if="chartErrors.trend" style="width:100%; height:350px; display:flex; align-items:center; justify-content:center;">
                            <el-empty description="数据加载失败" :image-size="80">
                                <el-button size="small" @click="loadTrendData">重新加载</el-button>
                            </el-empty>
                        </div>
                        <div v-else id="chartTrend" style="width:100%; height:350px;"></div>
                    </div>
                </el-col>
                <el-col :span="8">
                    <div class="page-card">
                        <div class="page-title">诈骗类型分布</div>
                        <div v-if="chartErrors.fraudType" style="width:100%; height:350px; display:flex; align-items:center; justify-content:center;">
                            <el-empty description="数据加载失败" :image-size="80">
                                <el-button size="small" @click="loadFraudTypeData">重新加载</el-button>
                            </el-empty>
                        </div>
                        <div v-else id="chartFraudType" style="width:100%; height:350px;"></div>
                    </div>
                </el-col>
            </el-row>

            <!-- 底部图表 -->
            <el-row :gutter="20" style="margin-top: 20px;">
                <el-col :span="12">
                    <div class="page-card" style="padding-bottom: 10px;">
                        <div class="page-title">24小时检测分布</div>
                        <div v-if="chartErrors.hourly" style="width:100%; height:260px; display:flex; align-items:center; justify-content:center;">
                            <el-empty description="数据加载失败" :image-size="60">
                                <el-button size="small" @click="loadHourlyData">重新加载</el-button>
                            </el-empty>
                        </div>
                        <div v-else id="chartHourly" style="width:100%; height:260px;"></div>
                    </div>
                </el-col>
                <el-col :span="12">
                    <div class="page-card" style="padding-bottom: 10px;">
                        <div class="page-title">风险等级分布</div>
                        <div id="chartRiskLevel" style="width:100%; height:260px;"></div>
                    </div>
                </el-col>
            </el-row>

            <!-- 最近检测记录 -->
            <el-row :gutter="20" style="margin-top: 20px;">
                <el-col :span="24">
                    <div class="page-card">
                        <div class="page-title">最近检测记录</div>
                        <div style="color: #6b7280; font-size: 13px; margin-top: 5px; margin-bottom: 15px;">
                            显示最近20条AI检测记录，包含风险评分和检测类型
                        </div>
                        <el-table :data="recentDetections" stripe style="margin-top: 15px; width: 100%;" :header-cell-style="{background:'#f8fafc', color:'#475569', fontWeight:'600'}">
                            <template #empty>
                                <el-empty description="暂无检测记录" :image-size="80" />
                            </template>
                            <el-table-column prop="log_id" label="ID" width="80" align="center"></el-table-column>
                            <el-table-column prop="call_id" label="通话ID" width="100" align="center"></el-table-column>
                            <el-table-column prop="caller_number" label="来电号码" width="150" align="center"></el-table-column>
                            <el-table-column prop="detection_type" label="检测类型" width="120" align="center">
                                <template #default="scope">
                                    <el-tag :type="getDetectionTypeType(scope.row.detection_type)">
                                        {{ getDetectionTypeLabel(scope.row.detection_type) }}
                                    </el-tag>
                                </template>
                            </el-table-column>
                            <el-table-column prop="overall_score" label="风险评分" width="180" align="center">
                                <template #default="scope">
                                    <el-progress 
                                        :percentage="scope.row.overall_score" 
                                        :color="getRiskColor(scope.row.overall_score)"
                                        :stroke-width="12"
                                        style="width: 140px; margin: 0 auto;">
                                    </el-progress>
                                </template>
                            </el-table-column>
                            <el-table-column prop="created_at" label="检测时间" align="center">
                                <template #default="scope">
                                    {{ formatTime(scope.row.created_at) }}
                                </template>
                            </el-table-column>
                        </el-table>
                    </div>
                </el-col>
            </el-row>
        </div>
    `,
    data() {
        return {
            loading: false,
            stats: {},
            trendDays: 7,
            trendData: [],
            fraudTypeData: [],
            hourlyData: { hours: [], counts: [] },
            recentDetections: [],
            systemHealth: {},
            charts: {},
            chartErrors: {
                trend: false,
                fraudType: false,
                hourly: false
            }
        }
    },
    async mounted() {
        await this.loadAllData();
        window.addEventListener('resize', this.handleResize);
    },
    beforeUnmount() {
        window.removeEventListener('resize', this.handleResize);
        // 销毁所有图表实例
        Object.values(this.charts).forEach(chart => {
            if (chart && !chart.isDisposed()) {
                chart.dispose();
            }
        });
        this.charts = {};
    },
    methods: {
        async loadAllData() {
            this.loading = true;
            try {
                await Promise.all([
                    this.loadStats(),
                    this.loadTrendData(),
                    this.loadFraudTypeData(),
                    this.loadHourlyData(),
                    this.loadRecentDetections(),
                    this.loadSystemHealth()
                ]);
                this.initCharts();
            } catch(e) { 
                console.error('加载数据失败:', e);
            } finally { 
                this.loading = false; 
            }
        },
        async loadStats() {
            this.stats = await api.getStats();
        },
        async loadTrendData() {
            try {
                this.chartErrors.trend = false;
                const data = await api.getTrendStats(this.trendDays);
                // 数据有效性检查
                if (!Array.isArray(data)) {
                    console.warn('趋势数据格式无效: 期望数组');
                    this.trendData = [];
                } else {
                    // 检查数组元素是否有必要字段
                    this.trendData = data.filter(item => 
                        item && typeof item.date !== 'undefined'
                    );
                }
                this.updateTrendChart();
            } catch (e) {
                console.error('加载趋势数据失败:', e);
                this.chartErrors.trend = true;
                this.trendData = [];
            }
        },
        async loadFraudTypeData() {
            try {
                this.chartErrors.fraudType = false;
                const data = await api.getFraudTypeStats();
                // 数据有效性检查
                if (!Array.isArray(data)) {
                    console.warn('诈骗类型数据格式无效: 期望数组');
                    this.fraudTypeData = [];
                } else {
                    // 检查数组元素是否有必要字段
                    this.fraudTypeData = data.filter(item => 
                        item && item.type && typeof item.value === 'number'
                    );
                }
                this.updateFraudTypeChart();
            } catch (e) {
                console.error('加载诈骗类型数据失败:', e);
                this.chartErrors.fraudType = true;
                this.fraudTypeData = [];
            }
        },
        async loadHourlyData() {
            try {
                this.chartErrors.hourly = false;
                const data = await api.getHourlyStats();
                // 数据有效性检查
                if (!data || typeof data !== 'object') {
                    console.warn('小时数据格式无效');
                    this.hourlyData = { hours: [], counts: [] };
                } else {
                    this.hourlyData = {
                        hours: Array.isArray(data.hours) ? data.hours : [],
                        counts: Array.isArray(data.counts) ? data.counts : []
                    };
                }
                this.updateHourlyChart();
            } catch (e) {
                console.error('加载小时数据失败:', e);
                this.chartErrors.hourly = true;
                this.hourlyData = { hours: [], counts: [] };
            }
        },
        async loadRecentDetections() {
            this.recentDetections = await api.getRecentDetections(10);
        },
        async loadSystemHealth() {
            this.systemHealth = await api.getSystemHealth();
        },
        async initCharts() {
            // 等待ECharts加载完成
            await this.waitForEcharts();
            
            // 使用 requestAnimationFrame 确保 DOM 已渲染
            requestAnimationFrame(() => {
                this.initTrendChart();
                this.initFraudTypeChart();
                this.initHourlyChart();
                this.initRiskLevelChart();
            });
        },
        waitForEcharts() {
            return new Promise((resolve) => {
                if (typeof echarts !== 'undefined') {
                    resolve();
                    return;
                }
                // 轮询检查echarts是否加载
                const checkInterval = setInterval(() => {
                    if (typeof echarts !== 'undefined') {
                        clearInterval(checkInterval);
                        resolve();
                    }
                }, 100);
                // 超时处理（10秒）
                setTimeout(() => {
                    clearInterval(checkInterval);
                    console.warn('ECharts加载超时');
                    resolve();
                }, 10000);
            });
        },
        initTrendChart() {
            const chartDom = document.getElementById('chartTrend');
            if (!chartDom || typeof echarts === 'undefined') return;
            
            // 确保容器有正确尺寸
            chartDom.style.width = '100%';
            chartDom.style.height = '300px';
            
            this.charts.trend = echarts.init(chartDom);
            
            // 初始化图表（updateTrendChart 内部会检查数据）
            this.updateTrendChart();
            
            // 初始化后强制调整大小
            setTimeout(() => this.charts.trend?.resize(), 100);
        },
        updateTrendChart() {
            if (!this.charts.trend) return;
            
            // 数据有效性检查
            if (!Array.isArray(this.trendData) || this.trendData.length === 0) {
                // 显示空状态
                this.charts.trend.setOption({
                    title: {
                        text: '暂无数据',
                        left: 'center',
                        top: 'center',
                        textStyle: { color: '#999', fontSize: 14 }
                    },
                    xAxis: { show: false },
                    yAxis: { show: false },
                    series: []
                });
                return;
            }
            
            const dates = this.trendData.map(d => d.date || '');
            const detections = this.trendData.map(d => d.detections || 0);
            const blocked = this.trendData.map(d => d.blocked || 0);
            const newUsers = this.trendData.map(d => d.new_users || 0);
            
            this.charts.trend.setOption({
                tooltip: { trigger: 'axis' },
                legend: { data: ['检测次数', '拦截次数', '新增用户'] },
                grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
                xAxis: { type: 'category', boundaryGap: false, data: dates, axisLabel: { interval: 0 } },
                yAxis: { type: 'value' },
                series: [
                    {
                        name: '检测次数',
                        type: 'line',
                        smooth: true,
                        data: detections,
                        itemStyle: { color: '#4f46e5' },
                        areaStyle: { color: 'rgba(79, 70, 229, 0.1)' }
                    },
                    {
                        name: '拦截次数',
                        type: 'line',
                        smooth: true,
                        data: blocked,
                        itemStyle: { color: '#dc2626' },
                        areaStyle: { color: 'rgba(220, 38, 38, 0.1)' }
                    },
                    {
                        name: '新增用户',
                        type: 'line',
                        smooth: true,
                        data: newUsers,
                        itemStyle: { color: '#059669' }
                    }
                ]
            });
        },
        initFraudTypeChart() {
            const chartDom = document.getElementById('chartFraudType');
            if (!chartDom || typeof echarts === 'undefined') return;
            
            chartDom.style.width = '100%';
            chartDom.style.height = '310px';
            
            this.charts.fraudType = echarts.init(chartDom);
            
            // 初始化图表（updateFraudTypeChart 内部有默认数据）
            this.updateFraudTypeChart();
            
            setTimeout(() => this.charts.fraudType?.resize(), 100);
        },
        updateFraudTypeChart() {
            if (!this.charts.fraudType) return;
            
            // 确保数据格式正确
            let data = this.fraudTypeData || [];
            data = data.filter(item => item && item.type && typeof item.value === 'number');
            
            if (data.length === 0) {
                data = [
                    { type: '文本诈骗', value: 35 },
                    { type: '音频伪造', value: 25 },
                    { type: '视频伪造', value: 20 },
                    { type: '图片诈骗', value: 20 }
                ];
            }
            
            this.charts.fraudType.setOption({
                tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
                legend: { 
                    orient: 'horizontal', 
                    bottom: '5%',
                    left: 'center'
                },
                series: [{
                    type: 'pie',
                    radius: ['35%', '60%'],
                    center: ['50%', '45%'],
                    avoidLabelOverlap: false,
                    itemStyle: {
                        borderRadius: 10,
                        borderColor: '#fff',
                        borderWidth: 2
                    },
                    label: { show: false, position: 'center' },
                    emphasis: {
                        label: { show: true, fontSize: 16, fontWeight: 'bold' }
                    },
                    data: data.map(d => ({ name: d.type, value: d.value }))
                }]
            });
        },
        initHourlyChart() {
            const chartDom = document.getElementById('chartHourly');
            if (!chartDom || typeof echarts === 'undefined') return;
            
            chartDom.style.width = '100%';
            chartDom.style.height = '260px';
            
            this.charts.hourly = echarts.init(chartDom);
            
            // 初始化图表（updateHourlyChart 内部有默认数据）
            this.updateHourlyChart();
            
            setTimeout(() => this.charts.hourly?.resize(), 100);
        },
        updateHourlyChart() {
            if (!this.charts.hourly) return;
            
            const hours = this.hourlyData.hours || Array.from({length: 24}, (_, i) => i + '时');
            const counts = this.hourlyData.counts || Array(24).fill(0);
            
            this.charts.hourly.setOption({
                tooltip: { trigger: 'axis' },
                grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
                xAxis: { type: 'category', data: hours },
                yAxis: { type: 'value' },
                series: [{
                    data: counts,
                    type: 'bar',
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: '#83bff6' },
                            { offset: 0.5, color: '#188df0' },
                            { offset: 1, color: '#188df0' }
                        ])
                    }
                }]
            });
        },
        initRiskLevelChart() {
            const chartDom = document.getElementById('chartRiskLevel');
            if (!chartDom || typeof echarts === 'undefined') return;
            
            chartDom.style.width = '100%';
            chartDom.style.height = '260px';
            
            this.charts.riskLevel = echarts.init(chartDom);
            
            setTimeout(() => this.charts.riskLevel?.resize(), 100);
            
            this.charts.riskLevel.setOption({
                tooltip: { 
                    trigger: 'item',
                    formatter: '{b}: {c}%'
                },
                legend: {
                    orient: 'horizontal',
                    bottom: '5%',
                    left: 'center',
                    itemWidth: 12,
                    itemHeight: 12,
                    textStyle: { fontSize: 12 }
                },
                series: [{
                    type: 'pie',
                    radius: ['45%', '75%'],
                    center: ['50%', '45%'],
                    label: {
                        show: false,
                        position: 'center'
                    },
                    emphasis: {
                        label: {
                            show: true,
                            formatter: '{b}\n{d}%',
                            fontSize: 16,
                            fontWeight: 'bold'
                        }
                    },
                    labelLine: { show: false },
                    data: [
                        { value: 15, name: '高危 (>80)', itemStyle: { color: '#dc2626' } },
                        { value: 25, name: '中危 (50-80)', itemStyle: { color: '#f59e0b' } },
                        { value: 60, name: '低危 (<50)', itemStyle: { color: '#10b981' } }
                    ]
                }]
            });
            
            window.addEventListener('resize', () => {
                this.charts.riskLevel?.resize();
            });
        },
        handleResize() {
            Object.values(this.charts).forEach(chart => chart && chart.resize());
        },
        destroyCharts() {
            Object.values(this.charts).forEach(chart => {
                if (chart) {
                    chart.dispose();
                }
            });
            this.charts = {};
        },
        getDetectionTypeType(type) {
            const map = { text: 'primary', audio: 'success', video: 'warning', image: 'info' };
            return map[type] || 'info';
        },
        getDetectionTypeLabel(type) {
            const map = { text: '文本', audio: '音频', video: '视频', image: '图片' };
            return map[type] || type;
        },
        getRiskColor(score) {
            if (score >= 80) return '#dc2626';
            if (score >= 50) return '#f59e0b';
            return '#10b981';
        },
        formatTime(time) {
            if (!time) return '-';
            return new Date(time).toLocaleString();
        },
        formatGrowthRate(rate) {
            if (rate === undefined || rate === null) return '--';
            const prefix = rate >= 0 ? '+' : '';
            return `${prefix}${rate.toFixed(1)}%`;
        }
    }
}