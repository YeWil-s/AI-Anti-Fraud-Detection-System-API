import api from '../js/api.js';

export default {
    template: `
        <div class="page-card">
            <div class="page-header">
                <div class="page-title">黑名单号码库</div>
                <el-button type="danger" @click="dialogVisible = true">拉黑号码</el-button>
            </div>
            
            <el-table :data="tableData" v-loading="loading">
                <el-table-column prop="id" label="ID" width="80"></el-table-column>
                <el-table-column prop="number" label="电话号码" width="180">
                     <template #default="{row}">
                        <b style="color:#d32f2f">{{ row.number }}</b>
                    </template>
                </el-table-column>
                <el-table-column prop="source" label="来源" width="120">
                    <template #default="{row}">
                        <el-tag type="info">{{ row.source }}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="created_at" label="拉黑时间"></el-table-column>
                <el-table-column label="操作">
                    <template #default="{row}">
                        <el-button size="small" @click="doDelete(row)">移出黑名单</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <el-dialog v-model="dialogVisible" title="手动拉黑" width="400px">
                <el-form :model="form">
                    <el-form-item label="号码">
                        <el-input v-model="form.number" placeholder="请输入完整电话号码" />
                    </el-form-item>
                    <el-form-item label="原因">
                        <el-input v-model="form.description" type="textarea" />
                    </el-form-item>
                </el-form>
                <template #footer>
                    <el-button @click="dialogVisible = false">取消</el-button>
                    <el-button type="danger" @click="doAdd">确认拉黑</el-button>
                </template>
            </el-dialog>
        </div>
    `,
    data() {
        return {
            loading: false, tableData: [], dialogVisible: false,
            form: { number: '', description: '', risk_level: 5, source: 'manual_admin' }
        }
    },
    mounted() { this.loadData(); },
    methods: {
        async loadData() {
            this.loading = true;
            this.tableData = await api.getBlacklist();
            this.loading = false;
        },
        async doAdd() {
            await api.addBlacklist(this.form);
            this.dialogVisible = false;
            this.loadData();
            ElementPlus.ElMessage.success('号码已拉黑');
        },
        async doDelete(row) {
            await api.delBlacklist(row.id);
            this.loadData();
            ElementPlus.ElMessage.success('已移出黑名单');
        }
    }
}