import api from '../js/api.js';

export default {
    template: `
        <div class="page-container">
            <div class="page-header" style="margin-bottom: 20px;">
                <h2>🤖 知识库进化与学习</h2>
                <p style="color: #666; margin-top: 10px;">在此处审核系统拦截的高危诈骗通话。点击“让AI学习”，系统将自动生成特征文件并吸收为永久防御经验。</p>
            </div>

            <el-card class="page-card" v-loading="loading">
                <el-table :data="cases" style="width: 100%" stripe border>
                    <el-table-column prop="call_id" label="通话ID" width="100" align="center"></el-table-column>
                    
                    <el-table-column prop="start_time" label="拦截时间" width="180">
                        <template #default="scope">
                            {{ scope.row.start_time ? new Date(scope.row.start_time).toLocaleString() : '未知' }}
                        </template>
                    </el-table-column>
                    
                    <el-table-column prop="target_number" label="恶意号码" width="150" align="center">
                        <template #default="scope">
                            <el-tag type="danger" effect="dark">{{ scope.row.target_number }}</el-tag>
                        </template>
                    </el-table-column>
                    
                    <el-table-column prop="details" label="AI检测摘要 / 诈骗话术" min-width="350">
                        <template #default="scope">
                            <div style="white-space: pre-wrap; word-break: break-all;">
                                {{ scope.row.details }}
                            </div>
                        </template>
                    </el-table-column>
                    
                    <el-table-column label="操作" width="160" align="center" fixed="right">
                        <template #default="scope">
                            <el-button 
                                type="primary" 
                                size="default" 
                                @click="handleLearn(scope.row)">
                                  学习此案例
                            </el-button>
                        </template>
                    </el-table-column>
                </el-table>
                
                <el-empty v-if="!loading && cases.length === 0" description="暂无需要学习的高危拦截案例"></el-empty>
            </el-card>
        </div>
    `,
    data() {
        return {
            loading: false,
            cases: []
        };
    },
    mounted() {
        this.fetchCases();
    },
    methods: {
        async fetchCases() {
            this.loading = true;
            try {
                // 请求后端获取高危列表
                this.cases = await api.getFraudCases();
            } catch (error) {
                console.error('获取案例失败:', error);
            } finally {
                this.loading = false;
            }
        },
        async handleLearn(row) {
            try {
                // 点击学习，调用我们在 admin.py 写的后端接口
                await api.learnCase(row.call_id);
                
                ElementPlus.ElMessage.success({
                    message: '✅ 案例特征文件已生成！系统将在今晚进行自主学习吸收。',
                    duration: 4000
                });
                
                // 学习成功后，为了界面清爽，可以将该条记录从当前列表中移除
                this.cases = this.cases.filter(item => item.call_id !== row.call_id);
            } catch (error) {
                // 错误提示已被 axios 拦截器处理
                console.error('加入学习队列失败:', error);
            }
        }
    }
};