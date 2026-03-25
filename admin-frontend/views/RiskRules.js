import api from '../js/api.js';

export default {
    template: `
        <div class="page-card">
            <div class="page-header">
                <div>
                    <div class="page-title">反诈关键词库</div>
                    <div style="color: #6b7280; font-size: 13px; margin-top: 5px;">
                        共 {{ tableData.length }} 条规则 | 阻断规则: {{ blockCount }} 条 | 告警规则: {{ alertCount }} 条
                    </div>
                </div>
                <el-button type="primary" @click="dialogVisible = true" size="large">
                    <i class="ri-add-line"></i> 新增规则
                </el-button>
            </div>

            <!-- 统计图表 -->
            <el-row :gutter="20" style="margin-bottom: 20px;">
                <el-col :span="12">
                    <div style="background: #f8fafc; border-radius: 8px; padding: 15px;">
                        <div style="font-weight: 600; margin-bottom: 10px;">风险等级分布</div>
                        <div id="riskLevelChart" style="height: 200px;"></div>
                    </div>
                </el-col>
                <el-col :span="12">
                    <div style="background: #f8fafc; border-radius: 8px; padding: 15px;">
                        <div style="font-weight: 600; margin-bottom: 10px;">动作类型分布</div>
                        <div id="actionChart" style="height: 200px;"></div>
                    </div>
                </el-col>
            </el-row>
            
            <!-- 搜索和操作栏 -->
            <div style="margin-bottom: 20px; display: flex; gap: 10px;">
                <el-input 
                    v-model="searchKeyword" 
                    placeholder="搜索关键词" 
                    style="width: 300px;"
                    clearable
                    @input="debouncedSearch">
                    <template #prefix>
                        <i class="ri-search-line"></i>
                    </template>
                </el-input>
                <el-select v-model="filterAction" placeholder="全部动作" clearable @change="handleSearch" style="width: 150px;">
                    <el-option label="阻断" value="block"></el-option>
                    <el-option label="告警" value="alert"></el-option>
                </el-select>
                <el-select v-model="filterLevel" placeholder="全部等级" clearable @change="handleSearch" style="width: 150px;">
                    <el-option label="1级" :value="1"></el-option>
                    <el-option label="2级" :value="2"></el-option>
                    <el-option label="3级" :value="3"></el-option>
                    <el-option label="4级" :value="4"></el-option>
                    <el-option label="5级" :value="5"></el-option>
                </el-select>
            </div>
            
            <el-table :data="filteredData" v-loading="loading" stripe border>
                <template #empty>
                    <el-empty description="暂无匹配的规则" :image-size="100" />
                </template>
                <el-table-column type="index" label="#" width="60" align="center"></el-table-column>
                <el-table-column prop="rule_id" label="ID" width="80" align="center"></el-table-column>
                <el-table-column prop="keyword" label="敏感词" min-width="150">
                     <template #default="{row}">
                        <el-tag effect="dark" size="large" style="font-size: 14px;">{{ row.keyword }}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column label="启用状态" width="100" align="center">
                    <template #default="{row}">
                        <el-switch
                            v-model="row.is_enabled"
                            @change="handleToggleRule(row)"
                            inline-prompt
                            active-text="启用"
                            inactive-text="禁用"
                            :loading="row._toggling"
                        />
                    </template>
                </el-table-column>
                <el-table-column prop="hit_count" label="命中次数" width="100" align="center">
                    <template #default="{row}">
                        <el-tag type="info" v-if="row.hit_count !== undefined">{{ row.hit_count }}</el-tag>
                        <span v-else style="color: #999;">--</span>
                    </template>
                </el-table-column>
                <el-table-column prop="risk_level" label="风险等级" width="180" align="center">
                    <template #default="{row}">
                        <el-rate v-model="row.risk_level" disabled show-score text-color="#ff9900"/>
                    </template>
                </el-table-column>
                <el-table-column prop="action" label="动作" width="120" align="center">
                    <template #default="{row}">
                        <el-tag :type="row.action=='block'?'danger':'warning'" size="large" effect="dark">
                            {{ row.action=='block'?'🚫 阻断':'⚠️ 告警' }}
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="description" label="描述" min-width="200">
                    <template #default="{row}">
                        {{ row.description || '-' }}
                    </template>
                </el-table-column>
                <el-table-column prop="created_at" label="创建时间" width="180">
                    <template #default="{row}">
                        {{ formatTime(row.created_at) }}
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="150" align="center" fixed="right">
                    <template #default="{row}">
                        <el-button link type="primary" @click="viewDetail(row)">详情</el-button>
                        <el-button link type="danger" @click="doDelete(row)">删除</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <!-- 新增规则对话框 -->
            <el-dialog v-model="dialogVisible" title="添加新规则" width="550px">
                <el-form :model="form" label-width="100px">
                    <el-form-item label="关键词" required>
                        <el-input v-model="form.keyword" placeholder="如：安全账户、转账汇款" />
                    </el-form-item>
                    <el-form-item label="风险等级">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <el-slider v-model="form.risk_level" :min="1" :max="5" show-stops style="flex: 1;" />
                            <el-tag type="danger" size="large">{{ form.risk_level }}级</el-tag>
                        </div>
                    </el-form-item>
                    <el-form-item label="动作">
                        <el-radio-group v-model="form.action">
                            <el-radio-button label="alert">⚠️ 告警</el-radio-button>
                            <el-radio-button label="block">🚫 阻断</el-radio-button>
                        </el-radio-group>
                    </el-form-item>
                    <el-form-item label="描述">
                        <el-input v-model="form.description" type="textarea" :rows="3" placeholder="规则描述（可选）" />
                    </el-form-item>
                </el-form>
                <template #footer>
                    <el-button @click="dialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="doAdd" :disabled="!form.keyword">提交</el-button>
                </template>
            </el-dialog>

            <!-- 详情对话框 -->
            <el-dialog v-model="detailVisible" title="规则详情" width="500px">
                <el-descriptions :column="1" border v-if="currentRule">
                    <el-descriptions-item label="规则ID">{{ currentRule.rule_id }}</el-descriptions-item>
                    <el-descriptions-item label="关键词">
                        <el-tag effect="dark" size="large">{{ currentRule.keyword }}</el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="风险等级">
                        <el-rate v-model="currentRule.risk_level" disabled show-score />
                    </el-descriptions-item>
                    <el-descriptions-item label="动作">
                        <el-tag :type="currentRule.action=='block'?'danger':'warning'" effect="dark">
                            {{ currentRule.action=='block'?'阻断':'告警' }}
                        </el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="命中次数">{{ currentRule.hit_count !== undefined ? currentRule.hit_count : '--' }}</el-descriptions-item>
                    <el-descriptions-item label="启用状态">
                        <el-tag :type="currentRule.is_enabled !== false ? 'success' : 'info'">
                            {{ currentRule.is_enabled !== false ? '已启用' : '已禁用' }}
                        </el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="描述">{{ currentRule.description || '无' }}</el-descriptions-item>
                    <el-descriptions-item label="创建时间">{{ formatTime(currentRule.created_at) }}</el-descriptions-item>
                    <el-descriptions-item label="更新时间">{{ formatTime(currentRule.updated_at) }}</el-descriptions-item>
                </el-descriptions>
            </el-dialog>
        </div>
    `,
    data() {
        return {
            loading: false,
            tableData: [],
            filteredData: [],
            dialogVisible: false,
            detailVisible: false,
            currentRule: null,
            searchKeyword: '',
            filterAction: '',
            filterLevel: '',
            form: { keyword: '', risk_level: 3, action: 'alert', description: '' },
            charts: {},
            searchTimer: null
        }
    },
    computed: {
        blockCount() {
            return this.tableData.filter(r => r.action === 'block').length;
        },
        alertCount() {
            return this.tableData.filter(r => r.action === 'alert').length;
        }
    },
    mounted() { 
        this.loadData();
        window.addEventListener('resize', this.handleResize);
    },
    beforeUnmount() {
        window.removeEventListener('resize', this.handleResize);
        Object.values(this.charts).forEach(c => c && c.dispose());
    },
    methods: {
        async loadData() {
            this.loading = true;
            try {
                const data = await api.getRules();
                // 为每条规则初始化 is_enabled 字段（如果后端没有提供，默认为 true）
                this.tableData = data.map(rule => ({
                    ...rule,
                    is_enabled: rule.is_enabled !== undefined ? rule.is_enabled : true,
                    _toggling: false
                }));
                this.filteredData = this.tableData;
                this.$nextTick(() => this.initCharts());
            } finally {
                this.loading = false;
            }
        },
        initCharts() {
            // 风险等级分布图
            const levelChart = echarts.init(document.getElementById('riskLevelChart'));
            this.charts.level = levelChart;
            
            const levelData = [0, 0, 0, 0, 0];
            this.tableData.forEach(r => {
                if (r.risk_level >= 1 && r.risk_level <= 5) {
                    levelData[r.risk_level - 1]++;
                }
            });
            
            levelChart.setOption({
                tooltip: { trigger: 'axis' },
                xAxis: { type: 'category', data: ['1级', '2级', '3级', '4级', '5级'] },
                yAxis: { type: 'value' },
                series: [{
                    data: levelData,
                    type: 'bar',
                    itemStyle: {
                        color: (params) => {
                            const colors = ['#10b981', '#34d399', '#fbbf24', '#f97316', '#dc2626'];
                            return colors[params.dataIndex];
                        },
                        borderRadius: [4, 4, 0, 0]
                    }
                }]
            });

            // 动作类型分布图
            const actionChart = echarts.init(document.getElementById('actionChart'));
            this.charts.action = actionChart;
            
            actionChart.setOption({
                tooltip: { trigger: 'item' },
                series: [{
                    type: 'pie',
                    radius: ['40%', '70%'],
                    data: [
                        { value: this.blockCount, name: '阻断', itemStyle: { color: '#dc2626' } },
                        { value: this.alertCount, name: '告警', itemStyle: { color: '#f59e0b' } }
                    ]
                }]
            });
        },
        handleResize() {
            Object.values(this.charts).forEach(c => c && c.resize());
        },
        debouncedSearch() {
            if (this.searchTimer) {
                clearTimeout(this.searchTimer);
            }
            this.searchTimer = setTimeout(() => {
                this.handleSearch();
            }, 300);
        },
        handleSearch() {
            this.filteredData = this.tableData.filter(row => {
                const matchKeyword = !this.searchKeyword || 
                    row.keyword.toLowerCase().includes(this.searchKeyword.toLowerCase());
                const matchAction = !this.filterAction || row.action === this.filterAction;
                const matchLevel = !this.filterLevel || row.risk_level === this.filterLevel;
                return matchKeyword && matchAction && matchLevel;
            });
        },
        async doAdd() {
            if (!this.form.keyword) return;
            try {
                await api.addRule(this.form);
                this.dialogVisible = false;
                this.form = { keyword: '', risk_level: 3, action: 'alert', description: '' };
                this.loadData();
                ElementPlus.ElMessage.success('添加成功');
            } catch (error) {
                console.error('添加失败:', error);
            }
        },
        async doDelete(row) {
            try {
                await ElementPlus.ElMessageBox.confirm('确认删除该规则?', '提示', { type: 'warning' });
                await api.delRule(row.rule_id);
                this.loadData();
                ElementPlus.ElMessage.success('删除成功');
            } catch (error) {
                if (error !== 'cancel') {
                    console.error('删除失败:', error);
                }
            }
        },
        viewDetail(row) {
            this.currentRule = row;
            this.detailVisible = true;
        },
        async handleToggleRule(row) {
            row._toggling = true;
            try {
                // 尝试调用后端API更新规则状态
                if (typeof api.updateRuleStatus === 'function') {
                    await api.updateRuleStatus(row.rule_id, row.is_enabled);
                    ElementPlus.ElMessage.success(row.is_enabled ? '规则已启用' : '规则已禁用');
                } else {
                    // 后端API不支持时，仅前端状态变更
                    ElementPlus.ElMessage.info(row.is_enabled ? '规则已启用（仅前端状态）' : '规则已禁用（仅前端状态）');
                }
            } catch (error) {
                // 失败时回滚状态
                row.is_enabled = !row.is_enabled;
                console.error('切换规则状态失败:', error);
                ElementPlus.ElMessage.error('操作失败');
            } finally {
                row._toggling = false;
            }
        },
        formatTime(time) {
            if (!time) return '-';
            return new Date(time).toLocaleString();
        }
    }
}