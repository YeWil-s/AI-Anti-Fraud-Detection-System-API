import api from '../js/api.js';

export default {
    template: `
        <div class="page-card">
            <div class="page-header">
                <div>
                    <div class="page-title">黑名单号码库</div>
                    <div style="color: #6b7280; font-size: 13px; margin-top: 5px;">
                        共 {{ tableData.length }} 个号码 | 手动添加: {{ manualCount }} | 系统拦截: {{ systemCount }}
                    </div>
                </div>
                <el-button type="danger" @click="dialogVisible = true" size="large">
                    <i class="ri-add-line"></i> 拉黑号码
                </el-button>
            </div>

            <!-- 统计卡片 -->
            <el-row :gutter="20" style="margin-bottom: 20px;">
                <el-col :span="8">
                    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); padding: 20px; border-radius: 12px; color: white;">
                        <div style="font-size: 32px; font-weight: bold;">{{ tableData.length }}</div>
                        <div style="font-size: 14px; opacity: 0.9;">黑名单总数</div>
                    </div>
                </el-col>
                <el-col :span="8">
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 12px; color: white;">
                        <div style="font-size: 32px; font-weight: bold;">{{ highRiskCount }}</div>
                        <div style="font-size: 14px; opacity: 0.9;">高风险号码</div>
                    </div>
                </el-col>
                <el-col :span="8">
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 12px; color: white;">
                        <div style="font-size: 32px; font-weight: bold;">{{ todayAdded }}</div>
                        <div style="font-size: 14px; opacity: 0.9;">今日新增</div>
                    </div>
                </el-col>
            </el-row>

            <!-- 搜索和操作栏 -->
            <div style="margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap;">
                <el-input 
                    v-model="searchNumber" 
                    placeholder="搜索电话号码" 
                    style="width: 300px;"
                    clearable
                    @input="debouncedSearch">
                    <template #prefix>
                        <i class="ri-search-line"></i>
                    </template>
                </el-input>
                <el-select v-model="filterSource" placeholder="全部来源" clearable @change="handleSearch" style="width: 150px;">
                    <el-option label="手动添加" value="manual_admin"></el-option>
                    <el-option label="系统拦截" value="system"></el-option>
                    <el-option label="用户举报" value="user_report"></el-option>
                </el-select>
                <el-select v-model="filterRisk" placeholder="风险等级" clearable @change="handleSearch" style="width: 150px;">
                    <el-option label="5级-极高" :value="5"></el-option>
                    <el-option label="4级-高" :value="4"></el-option>
                    <el-option label="3级-中" :value="3"></el-option>
                </el-select>
                <el-button 
                    type="danger" 
                    plain 
                    :disabled="selectedRows.length === 0"
                    @click="batchRemove"
                    :loading="batchRemoving">
                    <i class="ri-delete-bin-line"></i> 批量移出 ({{ selectedRows.length }})
                </el-button>
            </div>
            
            <el-table :data="filteredData" v-loading="loading" stripe border @selection-change="handleSelectionChange" ref="blacklistTable">
                <template #empty>
                    <el-empty description="暂无匹配的黑名单记录" :image-size="100" />
                </template>
                <el-table-column type="selection" width="55" />
                <el-table-column type="index" label="#" width="60" align="center"></el-table-column>
                <el-table-column prop="id" label="ID" width="80" align="center"></el-table-column>
                <el-table-column prop="number" label="电话号码" width="180" align="center">
                     <template #default="{row}">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
                            <i class="ri-phone-lock-line" style="color: #dc2626;"></i>
                            <b style="color:#dc2626; font-size: 16px;">{{ maskPhone(row.number) }}</b>
                        </div>
                    </template>
                </el-table-column>
                <el-table-column prop="risk_level" label="风险等级" width="120" align="center">
                    <template #default="{row}">
                        <el-tag :type="getRiskType(row.risk_level)" effect="dark">
                            {{ row.risk_level }}级
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="source" label="来源" width="140" align="center">
                    <template #default="{row}">
                        <el-tag :type="getSourceType(row.source)">
                            {{ getSourceLabel(row.source) }}
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="report_count" label="举报次数" width="100" align="center">
                    <template #default="{row}">
                        <el-badge :value="row.report_count || 0" :max="99" />
                    </template>
                </el-table-column>
                <el-table-column prop="description" label="拉黑原因" min-width="200">
                    <template #default="{row}">
                        {{ row.description || '无' }}
                    </template>
                </el-table-column>
                <el-table-column prop="created_at" label="拉黑时间" width="180">
                    <template #default="{row}">
                        {{ formatTime(row.created_at) }}
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="150" align="center" fixed="right">
                    <template #default="{row}">
                        <el-button link type="primary" @click="viewDetail(row)">详情</el-button>
                        <el-button link type="success" @click="doDelete(row)">移出</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <!-- 拉黑对话框 -->
            <el-dialog v-model="dialogVisible" title="手动拉黑号码" width="500px">
                <el-alert
                    title="注意事项"
                    description="请确保号码准确无误。拉黑后，该号码的所有来电将被自动拦截。"
                    type="warning"
                    show-icon
                    :closable="false"
                    style="margin-bottom: 20px;"
                />
                <el-form :model="form" :rules="formRules" ref="blacklistForm" label-width="100px">
                    <el-form-item label="电话号码" prop="number" required>
                        <el-input v-model="form.number" placeholder="请输入完整电话号码">
                            <template #prefix>
                                <i class="ri-phone-line"></i>
                            </template>
                        </el-input>
                    </el-form-item>
                    <el-form-item label="风险等级">
                        <el-rate v-model="form.risk_level" :max="5" show-score />
                    </el-form-item>
                    <el-form-item label="拉黑原因">
                        <el-input v-model="form.description" type="textarea" :rows="3" placeholder="请输入拉黑原因" />
                    </el-form-item>
                </el-form>
                <template #footer>
                    <el-button @click="dialogVisible = false">取消</el-button>
                    <el-button type="danger" @click="doAdd" :loading="adding">确认拉黑</el-button>
                </template>
            </el-dialog>

            <!-- 详情对话框 -->
            <el-dialog v-model="detailVisible" title="黑名单详情" width="500px">
                <el-descriptions :column="1" border v-if="currentItem">
                    <el-descriptions-item label="记录ID">{{ currentItem.id }}</el-descriptions-item>
                    <el-descriptions-item label="电话号码">
                        <b style="color: #dc2626; font-size: 18px;">{{ currentItem.number }}</b>
                    </el-descriptions-item>
                    <el-descriptions-item label="风险等级">
                        <el-rate v-model="currentItem.risk_level" disabled show-score />
                    </el-descriptions-item>
                    <el-descriptions-item label="来源">
                        <el-tag :type="getSourceType(currentItem.source)">
                            {{ getSourceLabel(currentItem.source) }}
                        </el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="举报次数">{{ currentItem.report_count || 0 }} 次</el-descriptions-item>
                    <el-descriptions-item label="拉黑原因">{{ currentItem.description || '无' }}</el-descriptions-item>
                    <el-descriptions-item label="拉黑时间">{{ formatTime(currentItem.created_at) }}</el-descriptions-item>
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
            currentItem: null,
            searchNumber: '',
            filterSource: '',
            filterRisk: '',
            form: { number: '', description: '', risk_level: 5, source: 'manual_admin' },
            formRules: {
                number: [
                    { required: true, message: '请输入电话号码', trigger: 'blur' },
                    { pattern: /^1[3-9]\d{9}$/, message: '请输入正确的手机号格式', trigger: 'blur' }
                ]
            },
            selectedRows: [],
            batchRemoving: false,
            adding: false,
            searchTimer: null
        }
    },
    computed: {
        manualCount() {
            return this.tableData.filter(i => i.source === 'manual_admin').length;
        },
        systemCount() {
            return this.tableData.filter(i => i.source === 'system').length;
        },
        highRiskCount() {
            return this.tableData.filter(i => i.risk_level >= 4).length;
        },
        todayAdded() {
            const today = new Date().toDateString();
            return this.tableData.filter(i => {
                if (!i.created_at) return false;
                return new Date(i.created_at).toDateString() === today;
            }).length;
        }
    },
    mounted() { this.loadData(); },
    methods: {
        handleSelectionChange(selection) {
            this.selectedRows = selection;
        },
        async batchRemove() {
            if (this.selectedRows.length === 0) return;
            
            try {
                await ElementPlus.ElMessageBox.confirm(
                    `确定将选中的 ${this.selectedRows.length} 个号码移出黑名单？`,
                    '批量移出确认',
                    { type: 'warning' }
                );
                
                this.batchRemoving = true;
                const promises = this.selectedRows.map(row => api.delBlacklist(row.id));
                await Promise.all(promises);
                
                ElementPlus.ElMessage.success(`成功移出 ${this.selectedRows.length} 个号码`);
                this.selectedRows = [];
                this.loadData();
            } catch (error) {
                if (error !== 'cancel') {
                    console.error('批量移出失败:', error);
                    ElementPlus.ElMessage.error('批量移出失败');
                }
            } finally {
                this.batchRemoving = false;
            }
        },
        maskPhone(phone) {
            if (!phone || phone.length < 7) return phone;
            return phone.substring(0, 3) + '****' + phone.substring(phone.length - 4);
        },
        debouncedSearch() {
            if (this.searchTimer) {
                clearTimeout(this.searchTimer);
            }
            this.searchTimer = setTimeout(() => {
                this.handleSearch();
            }, 300);
        },
        async loadData() {
            this.loading = true;
            try {
                this.tableData = await api.getBlacklist();
                this.filteredData = this.tableData;
            } finally {
                this.loading = false;
            }
        },
        handleSearch() {
            this.filteredData = this.tableData.filter(row => {
                const matchNumber = !this.searchNumber || 
                    row.number.includes(this.searchNumber);
                const matchSource = !this.filterSource || row.source === this.filterSource;
                const matchRisk = !this.filterRisk || row.risk_level === this.filterRisk;
                return matchNumber && matchSource && matchRisk;
            });
        },
        async doAdd() {
            this.$refs.blacklistForm.validate(async (valid) => {
                if (!valid) return;
                
                this.adding = true;
                try {
                    await api.addBlacklist(this.form);
                    this.dialogVisible = false;
                    this.form = { number: '', description: '', risk_level: 5, source: 'manual_admin' };
                    this.$refs.blacklistForm?.resetFields();
                    this.loadData();
                    ElementPlus.ElMessage.success('号码已拉黑');
                } catch (error) {
                    console.error('拉黑失败:', error);
                } finally {
                    this.adding = false;
                }
            });
        },
        async doDelete(row) {
            try {
                await ElementPlus.ElMessageBox.confirm('确定将该号码移出黑名单?', '提示', { type: 'warning' });
                await api.delBlacklist(row.id);
                this.loadData();
                ElementPlus.ElMessage.success('已移出黑名单');
            } catch (error) {
                if (error !== 'cancel') {
                    console.error('移出失败:', error);
                }
            }
        },
        viewDetail(row) {
            this.currentItem = row;
            this.detailVisible = true;
        },
        getRiskType(level) {
            if (level >= 4) return 'danger';
            if (level >= 3) return 'warning';
            return 'info';
        },
        getSourceType(source) {
            const map = { manual_admin: 'primary', system: 'danger', user_report: 'success' };
            return map[source] || 'info';
        },
        getSourceLabel(source) {
            const map = { manual_admin: '手动添加', system: '系统拦截', user_report: '用户举报' };
            return map[source] || source;
        },
        formatTime(time) {
            if (!time) return '-';
            return new Date(time).toLocaleString();
        }
    }
}