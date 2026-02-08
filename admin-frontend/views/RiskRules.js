import api from '../js/api.js';

export default {
    template: `
        <div class="page-card">
            <div class="page-header">
                <div class="page-title">反诈关键词库</div>
                <el-button type="primary" @click="dialogVisible = true">新增规则</el-button>
            </div>
            
            <el-table :data="tableData" v-loading="loading" stripe>
                <el-table-column prop="rule_id" label="ID" width="80"></el-table-column>
                <el-table-column prop="keyword" label="敏感词">
                     <template #default="{row}">
                        <el-tag effect="dark">{{ row.keyword }}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="risk_level" label="风险等级" width="150">
                    <template #default="{row}">
                        <el-rate v-model="row.risk_level" disabled show-score text-color="#ff9900"/>
                    </template>
                </el-table-column>
                <el-table-column prop="action" label="动作">
                    <template #default="{row}">
                        <el-tag :type="row.action=='block'?'danger':'warning'">{{ row.action=='block'?'阻断':'告警' }}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="120">
                    <template #default="{row}">
                        <el-button link type="danger" @click="doDelete(row)">删除</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <el-dialog v-model="dialogVisible" title="添加新规则" width="500px">
                <el-form :model="form">
                    <el-form-item label="关键词" label-width="80px">
                        <el-input v-model="form.keyword" placeholder="如：安全账户" />
                    </el-form-item>
                    <el-form-item label="等级" label-width="80px">
                        <el-slider v-model="form.risk_level" :min="1" :max="5" show-stops />
                    </el-form-item>
                    <el-form-item label="动作" label-width="80px">
                        <el-radio-group v-model="form.action">
                            <el-radio label="alert">告警</el-radio>
                            <el-radio label="block">阻断</el-radio>
                        </el-radio-group>
                    </el-form-item>
                </el-form>
                <template #footer>
                    <el-button @click="dialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="doAdd">提交</el-button>
                </template>
            </el-dialog>
        </div>
    `,
    data() {
        return {
            loading: false, tableData: [], dialogVisible: false,
            form: { keyword: '', risk_level: 3, action: 'alert' }
        }
    },
    mounted() { this.loadData(); },
    methods: {
        async loadData() {
            this.loading = true;
            this.tableData = await api.getRules();
            this.loading = false;
        },
        async doAdd() {
            if(!this.form.keyword) return;
            await api.addRule(this.form);
            this.dialogVisible = false;
            this.loadData();
            ElementPlus.ElMessage.success('添加成功');
        },
        async doDelete(row) {
            await ElementPlus.ElMessageBox.confirm('确认删除?');
            await api.delRule(row.rule_id);
            this.loadData();
        }
    }
}