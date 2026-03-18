import api from '../js/api.js';

export default {
    template: `
        <div class="page-card">
            <div class="page-header">
                <div>
                    <div class="page-title">系统规则测试台</div>
                    <div style="color: #6b7280; font-size: 13px; margin-top: 5px;">
                        实时测试风控规则匹配效果，验证关键词拦截逻辑
                    </div>
                </div>
            </div>

            <!-- 测试统计 -->
            <el-row :gutter="20" style="margin-bottom: 20px;">
                <el-col :span="6">
                    <div style="background: #f0f9ff; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #0369a1;">{{ testCount }}</div>
                        <div style="font-size: 12px; color: #64748b;">今日测试次数</div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div style="background: #fef2f2; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #dc2626;">{{ blockCount }}</div>
                        <div style="font-size: 12px; color: #64748b;">阻断次数</div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div style="background: #fffbeb; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #d97706;">{{ alertCount }}</div>
                        <div style="font-size: 12px; color: #64748b;">告警次数</div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #16a34a;">{{ passCount }}</div>
                        <div style="font-size: 12px; color: #64748b;">通过次数</div>
                    </div>
                </el-col>
            </el-row>

            <!-- 测试区域 -->
            <el-row :gutter="20">
                <el-col :span="14">
                    <div style="background:#fff; padding:20px; border-radius:8px; border:1px solid #e2e8f0;">
                        <div style="font-weight: 600; margin-bottom: 15px;">
                            <i class="ri-edit-line"></i> 输入测试文本
                        </div>
                        <el-input
                            v-model="inputText"
                            :rows="8"
                            type="textarea"
                            placeholder="请输入模拟诈骗文本，例如：'我是公安局的，请把钱转入安全账户'"
                        />
                        <div style="margin-top:15px; display: flex; gap: 10px;">
                            <el-button type="primary" @click="runTest" :loading="testing" size="large">
                                <i class="ri-flashlight-line"></i> 立即检测
                            </el-button>
                            <el-button @click="loadExample" size="large">
                                <i class="ri-file-copy-line"></i> 加载示例
                            </el-button>
                            <el-button @click="clearAll" size="large">
                                <i class="ri-delete-bin-line"></i> 清空
                            </el-button>
                        </div>
                    </div>
                </el-col>
                
                <el-col :span="10">
                    <div v-if="result" style="background:#f8fafc; padding:20px; border-radius:8px; border:1px solid #e2e8f0; height:100%">
                        <div style="margin-bottom:20px; font-weight:bold; font-size: 16px;">
                            <i class="ri-bar-chart-box-line"></i> 检测结果
                        </div>
                        
                        <div style="margin-bottom:20px;">
                            <div style="color:#64748b; margin-bottom:8px; font-size: 13px;">命中关键词</div>
                            <div>
                                <el-tag v-for="k in result.hit_keywords" :key="k" type="danger" effect="dark" style="margin-right:8px; margin-bottom:5px;">
                                    {{ k }}
                                </el-tag>
                                <el-tag v-if="result.hit_keywords.length===0" type="info">无命中</el-tag>
                            </div>
                        </div>

                        <div style="margin-bottom:20px;">
                            <div style="color:#64748b; margin-bottom:8px; font-size: 13px;">风险等级</div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <el-rate v-model="result.risk_level" disabled show-score text-color="#ff9900" />
                                <el-tag :type="getRiskType(result.risk_level)">{{ result.risk_level }}级风险</el-tag>
                            </div>
                        </div>

                        <div style="margin-bottom:20px;">
                            <div style="color:#64748b; margin-bottom:8px; font-size: 13px;">系统建议动作</div>
                            <el-tag size="large" effect="dark" :type="getActionType(result.action)">
                                <i :class="getActionIcon(result.action)"></i>
                                {{ getActionLabel(result.action) }}
                            </el-tag>
                        </div>

                        <div>
                            <div style="color:#64748b; margin-bottom:8px; font-size: 13px;">文本分析</div>
                            <div style="background: #fff; padding: 10px; border-radius: 4px; font-size: 13px;">
                                <div>文本长度: {{ result.text_length }} 字符</div>
                                <div>关键词密度: {{ ((result.hit_keywords.length / result.text_length) * 100).toFixed(2) }}%</div>
                            </div>
                        </div>
                    </div>
                    
                    <div v-else style="background:#f8fafc; padding:20px; border-radius:8px; border:1px solid #e2e8f0; height:100%; display: flex; align-items: center; justify-content: center; color: #94a3b8;">
                        <div style="text-align: center;">
                            <i class="ri-flashlight-line" style="font-size: 48px; margin-bottom: 10px;"></i>
                            <div>输入文本后点击检测</div>
                        </div>
                    </div>
                </el-col>
            </el-row>

            <!-- 测试历史 -->
            <div style="margin-top: 20px;" v-if="testHistory.length > 0">
                <div style="font-weight: 600; margin-bottom: 15px;">
                    <i class="ri-history-line"></i> 测试历史
                </div>
                <el-table :data="testHistory" stripe size="small" max-height="300">
                    <el-table-column type="index" label="#" width="50"></el-table-column>
                    <el-table-column prop="text" label="测试文本" min-width="300">
                        <template #default="scope">
                            <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 300px;">
                                {{ scope.row.text }}
                            </div>
                        </template>
                    </el-table-column>
                    <el-table-column prop="risk_level" label="风险等级" width="120" align="center">
                        <template #default="scope">
                            <el-rate v-model="scope.row.risk_level" disabled />
                        </template>
                    </el-table-column>
                    <el-table-column prop="action" label="动作" width="100" align="center">
                        <template #default="scope">
                            <el-tag :type="getActionType(scope.row.action)" size="small">
                                {{ scope.row.action }}
                            </el-tag>
                        </template>
                    </el-table-column>
                    <el-table-column prop="hit_count" label="命中数" width="80" align="center">
                        <template #default="scope">
                            <el-tag type="danger" size="small">{{ scope.row.hit_keywords.length }}</el-tag>
                        </template>
                    </el-table-column>
                    <el-table-column prop="time" label="时间" width="100">
                        <template #default="scope">
                            {{ formatTime(scope.row.time) }}
                        </template>
                    </el-table-column>
                </el-table>
            </div>
        </div>
    `,
    data() {
        return {
            inputText: '',
            testing: false,
            result: null,
            testHistory: [],
            testCount: 0,
            blockCount: 0,
            alertCount: 0,
            passCount: 0,
            examples: [
                "我是公安局的，你涉嫌洗钱犯罪，请把钱转入安全账户配合调查",
                "您好，您的快递已到达，请点击链接领取",
                "恭喜您中奖了，请提供银行卡信息领取奖金",
                "我是你领导，现在急需用钱，快给我转账"
            ]
        }
    },
    methods: {
        async runTest() {
            if (!this.inputText) {
                ElementPlus.ElMessage.warning('请输入测试文本');
                return;
            }
            this.testing = true;
            try {
                this.result = await api.testTextMatch(this.inputText);
                
                // 添加到历史记录
                this.testHistory.unshift({
                    text: this.inputText,
                    risk_level: this.result.risk_level,
                    action: this.result.action,
                    hit_keywords: this.result.hit_keywords,
                    time: new Date()
                });
                
                // 限制历史记录数量
                if (this.testHistory.length > 20) {
                    this.testHistory = this.testHistory.slice(0, 20);
                }
                
                // 更新统计
                this.testCount++;
                if (this.result.action === 'block') this.blockCount++;
                else if (this.result.action === 'alert') this.alertCount++;
                else this.passCount++;
                
                ElementPlus.ElMessage.success('检测完成');
            } catch (error) {
                console.error('检测失败:', error);
            } finally {
                this.testing = false;
            }
        },
        loadExample() {
            const randomExample = this.examples[Math.floor(Math.random() * this.examples.length)];
            this.inputText = randomExample;
        },
        clearAll() {
            this.inputText = '';
            this.result = null;
        },
        getRiskType(level) {
            if (level >= 4) return 'danger';
            if (level >= 3) return 'warning';
            return 'success';
        },
        getActionType(action) {
            const map = { block: 'danger', alert: 'warning', pass: 'success' };
            return map[action] || 'info';
        },
        getActionLabel(action) {
            const map = { block: '阻断拦截', alert: '风险告警', pass: '正常通过' };
            return map[action] || action;
        },
        getActionIcon(action) {
            const map = { 
                block: 'ri-shield-cross-line', 
                alert: 'ri-alarm-warning-line', 
                pass: 'ri-check-line' 
            };
            return map[action] || 'ri-question-line';
        },
        formatTime(time) {
            if (!time) return '-';
            return new Date(time).toLocaleTimeString();
        }
    }
}