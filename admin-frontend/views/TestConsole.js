import api from '../js/api.js';

export default {
    template: `
        <div class="page-card">
            <div class="page-title" style="margin-bottom:20px;">系统规则测试台</div>
            <el-alert title="此处测试直接调用后端匹配引擎，请先在规则库中添加关键词" type="info" show-icon style="margin-bottom:20px;"></el-alert>

            <el-row :gutter="20">
                <el-col :span="12">
                    <el-input
                        v-model="inputText"
                        :rows="6"
                        type="textarea"
                        placeholder="请输入模拟诈骗文本，例如：'我是公安局的，请把钱转入安全账户'"
                    />
                    <div style="margin-top:15px;">
                        <el-button type="primary" @click="runTest" :loading="testing">立即检测</el-button>
                    </div>
                </el-col>
                
                <el-col :span="12">
                    <div v-if="result" style="background:#f8fafc; padding:20px; border-radius:8px; border:1px solid #e2e8f0; height:100%">
                        <div style="margin-bottom:10px; font-weight:bold;">检测结果：</div>
                        
                        <div style="margin-bottom:10px;">
                            <span style="color:#64748b">命中关键词：</span>
                            <div style="margin-top:5px;">
                                <el-tag v-for="k in result.hit_keywords" :key="k" type="danger" style="margin-right:5px">{{ k }}</el-tag>
                                <span v-if="result.hit_keywords.length===0" style="color:#94a3b8">无命中</span>
                            </div>
                        </div>

                        <div style="margin-bottom:10px;">
                            <span style="color:#64748b">风险等级：</span>
                            <el-rate v-model="result.risk_level" disabled show-score text-color="#ff9900" />
                        </div>

                        <div>
                            <span style="color:#64748b">系统建议动作：</span>
                            <el-tag effect="dark" :type="result.action==='block'?'danger':(result.action==='alert'?'warning':'success')">
                                {{ result.action.toUpperCase() }}
                            </el-tag>
                        </div>
                    </div>
                </el-col>
            </el-row>
        </div>
    `,
    data() {
        return { inputText: '', testing: false, result: null }
    },
    methods: {
        async runTest() {
            if(!this.inputText) return;
            this.testing = true;
            try {
                // 真实调用后端 API
                this.result = await api.testTextMatch(this.inputText);
                ElementPlus.ElMessage.success('检测完成');
            } finally {
                this.testing = false;
            }
        }
    }
}