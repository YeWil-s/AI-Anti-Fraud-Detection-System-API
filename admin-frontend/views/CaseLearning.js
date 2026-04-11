import api from '../js/api.js';

export default {
    template: `
        <div class="page-container">
            <div class="page-header" style="margin-bottom: 20px;">
                <h2>AI 知识库进化中心</h2>
                <p style="color: #666; margin-top: 10px;">管理诈骗案例学习流程：上传新案例、审核系统拦截记录、追踪学习进度</p>
            </div>

            <!-- 统计卡片 -->
            <el-row :gutter="20" style="margin-bottom: 20px;">
                <el-col :span="6">
                    <div style="background: #f0f9ff; padding: 20px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 32px; font-weight: bold; color: #0369a1;">{{ pendingFiles.length }}</div>
                        <div style="font-size: 14px; color: #64748b;">待学习案例</div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div style="background: #f0fdf4; padding: 20px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 32px; font-weight: bold; color: #16a34a;">{{ learnedFiles.length }}</div>
                        <div style="font-size: 14px; color: #64748b;">已学习案例</div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div style="background: #fef2f2; padding: 20px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 32px; font-weight: bold; color: #dc2626;">{{ cases.length }}</div>
                        <div style="font-size: 14px; color: #64748b;">高危拦截记录</div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <el-button type="primary" size="large" @click="activeTab = 'upload'" style="width: 100%; height: 100%;">
                        <i class="ri-upload-cloud-line"></i> 上传新案例
                    </el-button>
                </el-col>
            </el-row>

            <!-- 标签页 -->
            <el-tabs v-model="activeTab" type="border-card">
                <!-- 案例上传 -->
                <el-tab-pane label="案例上传" name="upload">
                    <div style="padding: 20px;">
                        <el-alert
                            title="上传案例说明"
                            description="支持手动填写或上传文件(txt/json)。上传文件后系统会自动识别内容，您只需补全缺失的属性即可。"
                            type="info"
                            show-icon
                            :closable="false"
                            style="margin-bottom: 20px;"
                        />
                        
                        <!-- 文件上传区域 -->
                        <div style="margin-bottom: 20px;">
                            <el-upload
                                drag
                                action="#"
                                :auto-upload="false"
                                :on-change="handleFileChange"
                                :limit="1"
                                accept=".txt,.json"
                                style="width: 100%;">
                                <el-icon style="font-size: 48px; color: #409EFF;"><i class="ri-upload-cloud-line"></i></el-icon>
                                <div style="margin-top: 10px;">拖拽文件到此处或 <em>点击上传</em></div>
                                <template #tip>
                                    <div style="font-size: 12px; color: #909399; margin-top: 10px;">
                                        支持 .txt 和 .json 格式文件
                                    </div>
                                </template>
                            </el-upload>
                        </div>
                        
                        <el-form :model="uploadForm" label-width="100px" :rules="uploadRules" ref="uploadFormRef">
                            <div style="margin-bottom: 16px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <div style="font-weight:600;">文件属性识别结果</div>
                                    <el-button size="small" type="primary" plain @click="autoFillByLLM" :loading="llmFilling">
                                        LLM一键补全
                                    </el-button>
                                </div>
                                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                                    绿色为文件中已识别字段，灰色为缺失字段（可由你手动或LLM补全）。
                                </div>
                                <div style="display:flex; flex-wrap:wrap; gap:8px;">
                                    <el-tag v-for="item in parsedFieldPreview" :key="item.key" :type="item.filled ? 'success' : 'info'">
                                        {{ item.label }}：{{ item.filled ? '已识别' : '缺失' }}
                                    </el-tag>
                                </div>
                            </div>
                            <el-row :gutter="20">
                                <el-col :span="12">
                                    <el-form-item label="诈骗类型" prop="fraud_type">
                                        <el-select v-model="uploadForm.fraud_type" placeholder="选择诈骗类型" style="width: 100%;">
                                            <el-option label="1. 刷单返利诈骗" value="刷单返利诈骗"></el-option>
                                            <el-option label="2. 虚假投资理财诈骗" value="虚假投资理财诈骗"></el-option>
                                            <el-option label="3. 冒充客服诈骗" value="冒充客服诈骗"></el-option>
                                            <el-option label="4. 冒充公检法诈骗" value="冒充公检法诈骗"></el-option>
                                            <el-option label="5. 杀猪盘网恋诈骗" value="杀猪盘网恋诈骗"></el-option>
                                            <el-option label="6. 虚假贷款诈骗" value="虚假贷款诈骗"></el-option>
                                            <el-option label="7. 冒充领导熟人诈骗" value="冒充领导熟人诈骗"></el-option>
                                            <el-option label="8. 游戏产品虚假交易" value="游戏产品虚假交易"></el-option>
                                            <el-option label="9. 婚恋交友诈骗" value="婚恋交友诈骗"></el-option>
                                            <el-option label="10. 消除不良记录诈骗" value="消除不良记录诈骗"></el-option>
                                            <el-option label="其他" value="其他"></el-option>
                                        </el-select>
                                    </el-form-item>
                                </el-col>
                                <el-col :span="12">
                                    <el-form-item label="内容模态" prop="modality">
                                        <el-select v-model="uploadForm.modality" placeholder="选择内容类型" style="width: 100%;">
                                            <el-option label="文本对话" value="text"></el-option>
                                            <el-option label="音频录音" value="audio"></el-option>
                                            <el-option label="视频通话" value="video"></el-option>
                                            <el-option label="聊天截图" value="image"></el-option>
                                        </el-select>
                                    </el-form-item>
                                </el-col>
                            </el-row>
                            
                            <el-row :gutter="20">
                                <el-col :span="12">
                                    <el-form-item label="风险等级" prop="risk_level">
                                        <el-radio-group v-model="uploadForm.risk_level">
                                            <el-radio-button label="高危">高危</el-radio-button>
                                            <el-radio-button label="中危">中危</el-radio-button>
                                            <el-radio-button label="低危">低危</el-radio-button>
                                        </el-radio-group>
                                    </el-form-item>
                                </el-col>
                                <el-col :span="12">
                                    <el-form-item label="标签">
                                        <el-select
                                            v-model="uploadForm.tags"
                                            multiple
                                            filterable
                                            allow-create
                                            default-first-option
                                            placeholder="输入标签"
                                            style="width: 100%;">
                                            <el-option label="紧急" value="紧急"></el-option>
                                            <el-option label="新型" value="新型"></el-option>
                                            <el-option label="高发" value="高发"></el-option>
                                            <el-option label="跨境" value="跨境"></el-option>
                                        </el-select>
                                    </el-form-item>
                                </el-col>
                            </el-row>
                            
                            <el-form-item label="案例内容" prop="content">
                                <el-input
                                    v-model="uploadForm.content"
                                    type="textarea"
                                    :rows="6"
                                    placeholder="请输入诈骗案例的详细内容，如对话记录、话术文本等..."
                                />
                            </el-form-item>
                            
                            <el-form-item label="来源">
                                <el-input v-model="uploadForm.source" placeholder="案例来源（如：用户举报、新闻报道等）" />
                            </el-form-item>
                            
                            <el-form-item>
                                <el-button type="primary" @click="submitUpload" :loading="uploading">
                                    提交到待处理库
                                </el-button>
                                <el-button @click="resetUploadForm">重置</el-button>
                            </el-form-item>
                        </el-form>
                    </div>
                </el-tab-pane>

                <!-- 待学习案例 -->
                <el-tab-pane label="待学习案例" name="pending">
                    <el-table :data="pendingFiles" v-loading="loadingPending" stripe>
                        <el-table-column prop="filename" label="文件名" min-width="250"></el-table-column>
                        <el-table-column prop="case_count" label="案例数量" width="100" align="center">
                            <template #default="scope">
                                <el-tag type="primary">{{ scope.row.case_count }} 条</el-tag>
                            </template>
                        </el-table-column>
                        <el-table-column prop="size" label="文件大小" width="120">
                            <template #default="scope">
                                {{ formatFileSize(scope.row.size) }}
                            </template>
                        </el-table-column>
                        <el-table-column prop="modified" label="修改时间" width="180">
                            <template #default="scope">
                                {{ formatTime(scope.row.modified) }}
                            </template>
                        </el-table-column>
                        <el-table-column label="操作" width="200" fixed="right">
                            <template #default="scope">
                                <el-button link type="primary" @click="viewPendingDetail(scope.row)">查看</el-button>
                                <el-button link type="danger" @click="deletePendingFile(scope.row)">删除</el-button>
                            </template>
                        </el-table-column>
                    </el-table>
                    <el-empty v-if="!loadingPending && pendingFiles.length === 0" description="暂无待学习案例"></el-empty>
                </el-tab-pane>

                <!-- 已学习案例 -->
                <el-tab-pane label="已学习案例" name="learned">
                    <el-table :data="learnedFiles" v-loading="loadingLearned" stripe>
                        <el-table-column prop="filename" label="文件名" min-width="300"></el-table-column>
                        <el-table-column prop="size" label="文件大小" width="120">
                            <template #default="scope">
                                {{ formatFileSize(scope.row.size) }}
                            </template>
                        </el-table-column>
                        <el-table-column prop="learned_at" label="学习时间" width="180">
                            <template #default="scope">
                                {{ formatTime(scope.row.learned_at) }}
                            </template>
                        </el-table-column>
                        <el-table-column label="状态" width="100" align="center">
                            <template #default="scope">
                                <el-tag type="success">已入库</el-tag>
                            </template>
                        </el-table-column>
                    </el-table>
                    <el-empty v-if="!loadingLearned && learnedFiles.length === 0" description="暂无已学习案例"></el-empty>
                </el-tab-pane>

                <!-- 高危拦截审核 -->
                <el-tab-pane label="高危拦截审核" name="review">
                    <el-alert
                        title="审核说明"
                        description="点击“编辑并提交”按钮，对系统拦截的高危通话进行标注。提交后将写入待处理案例库。"
                        type="info"
                        show-icon
                        :closable="false"
                        style="margin-bottom: 15px;"
                    />
                    <el-table :data="cases" v-loading="loading" stripe border>
                        <el-table-column prop="call_id" label="通话ID" width="80" align="center"></el-table-column>
                        <el-table-column prop="start_time" label="拦截时间" width="160">
                            <template #default="scope">
                                {{ scope.row.start_time ? new Date(scope.row.start_time).toLocaleString() : '未知' }}
                            </template>
                        </el-table-column>
                        <el-table-column prop="target_number" label="来电号码" width="130" align="center">
                            <template #default="scope">
                                <el-tag type="danger" effect="dark" size="small">{{ scope.row.target_number }}</el-tag>
                            </template>
                        </el-table-column>
                        <el-table-column prop="fraud_type" label="诈骗类型" width="150" align="center">
                            <template #default="scope">
                                <el-tag v-if="scope.row.fraud_type && scope.row.fraud_type !== '未分类拦截'" type="warning">
                                    {{ scope.row.fraud_type }}
                                </el-tag>
                                <span v-else style="color: #999;">-</span>
                            </template>
                        </el-table-column>
                        <el-table-column prop="details" label="AI检测摘要" min-width="280">
                            <template #default="scope">
                                <div style="white-space: pre-wrap; word-break: break-all; max-height: 80px; overflow-y: auto;">{{ scope.row.details }}</div>
                            </template>
                        </el-table-column>
                        <el-table-column label="操作" width="140" align="center" fixed="right">
                            <template #default="scope">
                                <el-button type="primary" size="small" @click="openEditDialog(scope.row)">
                                    编辑并提交
                                </el-button>
                            </template>
                        </el-table-column>
                    </el-table>
                    <el-empty v-if="!loading && cases.length === 0" description="暂无需要审核的高危拦截案例"></el-empty>
                </el-tab-pane>
            </el-tabs>

            <!-- 案例详情对话框 -->
            <el-dialog v-model="detailDialogVisible" title="案例详情" width="900px">
                <el-table
                    v-if="currentDetail && currentDetail.length > 0"
                    :data="currentDetail"
                    border
                    stripe
                    max-height="520">
                    <el-table-column type="index" label="#" width="55" align="center"></el-table-column>
                    <el-table-column prop="modality" label="模态" width="90" align="center"></el-table-column>
                    <el-table-column prop="fraud_type" label="诈骗类型" width="150" align="center"></el-table-column>
                    <el-table-column prop="risk_level" label="风险等级" width="100" align="center">
                        <template #default="scope">
                            <el-tag :type="scope.row.risk_level === '高危' ? 'danger' : (scope.row.risk_level === '中危' ? 'warning' : 'success')">
                                {{ scope.row.risk_level || '-' }}
                            </el-tag>
                        </template>
                    </el-table-column>
                    <el-table-column prop="source" label="来源" min-width="180"></el-table-column>
                    <el-table-column label="标签" width="170">
                        <template #default="scope">
                            <el-tag
                                v-for="tag in (Array.isArray(scope.row.tags) ? scope.row.tags : [])"
                                :key="tag"
                                size="small"
                                style="margin: 2px 4px 2px 0;">
                                {{ tag }}
                            </el-tag>
                            <span v-if="!Array.isArray(scope.row.tags) || scope.row.tags.length === 0" style="color:#999;">-</span>
                        </template>
                    </el-table-column>
                    <el-table-column prop="content" label="内容" min-width="280">
                        <template #default="scope">
                            <div style="white-space: pre-wrap; line-height: 1.5;">{{ scope.row.content || '-' }}</div>
                        </template>
                    </el-table-column>
                    <el-table-column prop="uploaded_at" label="上传时间" width="170">
                        <template #default="scope">
                            {{ formatTime(scope.row.uploaded_at) }}
                        </template>
                    </el-table-column>
                    <el-table-column prop="uploader" label="上传者" width="90" align="center"></el-table-column>
                </el-table>
                <el-empty v-else description="暂无案例详情"></el-empty>
            </el-dialog>

            <!-- 编辑并提交对话框 -->
            <el-dialog v-model="editDialogVisible" title="编辑案例并提交到待处理库" width="650px">
                <el-form :model="editForm" label-width="100px" :rules="editRules" ref="editFormRef">
                    <el-form-item label="通话ID">
                        <el-input v-model="editForm.call_id" disabled />
                    </el-form-item>
                    <el-form-item label="来电号码">
                        <el-input v-model="editForm.target_number" disabled />
                    </el-form-item>
                    <el-form-item label="诈骗类型" prop="fraud_type">
                        <el-select v-model="editForm.fraud_type" placeholder="请选择诈骗类型" style="width: 100%;">
                            <el-option label="1. 刷单返利诈骗" value="刷单返利诈骗"></el-option>
                            <el-option label="2. 虚假投资理财诈骗" value="虚假投资理财诈骗"></el-option>
                            <el-option label="3. 冒充客服诈骗" value="冒充客服诈骗"></el-option>
                            <el-option label="4. 冒充公检法诈骗" value="冒充公检法诈骗"></el-option>
                            <el-option label="5. 杀猪盘网恋诈骗" value="杀猪盘网恋诈骗"></el-option>
                            <el-option label="6. 虚假贷款诈骗" value="虚假贷款诈骗"></el-option>
                            <el-option label="7. 冒充领导熟人诈骗" value="冒充领导熟人诈骗"></el-option>
                            <el-option label="8. 游戏产品虚假交易" value="游戏产品虚假交易"></el-option>
                            <el-option label="9. 婚恋交友诈骗" value="婚恋交友诈骗"></el-option>
                            <el-option label="10. 消除不良记录诈骗" value="消除不良记录诈骗"></el-option>
                            <el-option label="其他" value="其他"></el-option>
                        </el-select>
                    </el-form-item>
                    <el-form-item label="自定义类型" v-if="editForm.fraud_type === '其他'">
                        <el-input v-model="editForm.custom_fraud_type" placeholder="请输入其他诈骗类型" />
                    </el-form-item>
                    <el-form-item label="内容模态">
                        <el-radio-group v-model="editForm.modality">
                            <el-radio-button label="text">文本</el-radio-button>
                            <el-radio-button label="audio">音频</el-radio-button>
                            <el-radio-button label="video">视频</el-radio-button>
                        </el-radio-group>
                    </el-form-item>
                    <el-form-item label="风险等级">
                        <el-radio-group v-model="editForm.risk_level">
                            <el-radio-button label="高危">高危</el-radio-button>
                            <el-radio-button label="中危">中危</el-radio-button>
                            <el-radio-button label="低危">低危</el-radio-button>
                        </el-radio-group>
                    </el-form-item>
                    <el-form-item label="AI摘要">
                        <el-input 
                            v-model="editForm.details" 
                            type="textarea" 
                            :rows="4"
                            placeholder="AI生成的检测摘要，可手动编辑..."
                        />
                    </el-form-item>
                    <el-form-item label="完整内容">
                        <div style="display:flex; justify-content:flex-end; margin-bottom: 8px;">
                            <el-button
                                size="small"
                                type="primary"
                                plain
                                :loading="rewriteLoading"
                                :disabled="!(editForm.content || editForm.details)"
                                @click="rewriteEditContentByLLM">
                                LLM一键改写为案例
                            </el-button>
                        </div>
                        <el-input 
                            v-model="editForm.content" 
                            type="textarea" 
                            :rows="6"
                            placeholder="案例完整内容（建议为案例叙事体，仅描述发生了什么）..."
                        />
                    </el-form-item>
                    <el-form-item label="标签">
                        <el-select
                            v-model="editForm.tags"
                            multiple
                            filterable
                            allow-create
                            default-first-option
                            placeholder="输入标签"
                            style="width: 100%;">
                            <el-option label="系统拦截" value="系统拦截"></el-option>
                            <el-option label="高危" value="高危"></el-option>
                            <el-option label="新型手法" value="新型手法"></el-option>
                            <el-option label="跨境" value="跨境"></el-option>
                        </el-select>
                    </el-form-item>
                </el-form>
                <template #footer>
                    <el-button @click="editDialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="submitEditLearn" :loading="editLoading">
                        确认并提交
                    </el-button>
                </template>
            </el-dialog>
        </div>
    `,
    data() {
        return {
            activeTab: 'upload',
            loading: false,
            loadingPending: false,
            loadingLearned: false,
            uploading: false,
            llmFilling: false,
            editLoading: false,
            rewriteLoading: false,
            cases: [],
            pendingFiles: [],
            learnedFiles: [],
            showUploadDialog: false,
            detailDialogVisible: false,
            editDialogVisible: false,
            currentDetail: null,
            currentCase: null,
            uploadForm: {
                modality: 'text',
                fraud_type: '',
                risk_level: '高危',
                content: '',
                source: '',
                tags: [],
                uploader: 'admin'
            },
            parsedFieldPreview: [
                { key: 'fraud_type', label: '诈骗类型', filled: false },
                { key: 'modality', label: '内容模态', filled: false },
                { key: 'risk_level', label: '风险等级', filled: false },
                { key: 'content', label: '案例内容', filled: false },
                { key: 'source', label: '来源', filled: false },
                { key: 'tags', label: '标签', filled: false }
            ],
            uploadRules: {
                fraud_type: [{ required: true, message: '请选择诈骗类型', trigger: 'change' }],
                modality: [{ required: true, message: '请选择内容模态', trigger: 'change' }],
                risk_level: [{ required: true, message: '请选择风险等级', trigger: 'change' }],
                content: [{ required: true, message: '请输入案例内容', trigger: 'blur' }]
            },
            editForm: {
                call_id: '',
                target_number: '',
                fraud_type: '',
                custom_fraud_type: '',
                modality: 'audio',
                risk_level: '高危',
                details: '',
                content: '',
                tags: ['系统拦截', '高危']
            },
            editRules: {
                fraud_type: [{ required: true, message: '请选择诈骗类型', trigger: 'change' }]
            },
            fraudTypeOptions: [
                '刷单返利诈骗',
                '虚假投资理财诈骗',
                '冒充客服诈骗',
                '冒充公检法诈骗',
                '杀猪盘网恋诈骗',
                '虚假贷款诈骗',
                '冒充领导熟人诈骗',
                '游戏产品虚假交易',
                '婚恋交友诈骗',
                '消除不良记录诈骗'
            ]
        };
    },
    mounted() {
        this.fetchAllData();
    },
    methods: {
        async fetchAllData() {
            await Promise.all([
                this.fetchCases(),
                this.fetchPendingFiles(),
                this.fetchLearnedFiles()
            ]);
        },
        async fetchCases() {
            this.loading = true;
            try {
                this.cases = await api.getFraudCases();
            } catch (error) {
                console.error('获取案例失败:', error);
            } finally {
                this.loading = false;
            }
        },
        async fetchPendingFiles() {
            this.loadingPending = true;
            try {
                this.pendingFiles = await api.getPendingCases();
            } catch (error) {
                console.error('获取待学习案例失败:', error);
            } finally {
                this.loadingPending = false;
            }
        },
        async fetchLearnedFiles() {
            this.loadingLearned = true;
            try {
                this.learnedFiles = await api.getLearnedCases();
            } catch (error) {
                console.error('获取已学习案例失败:', error);
            } finally {
                this.loadingLearned = false;
            }
        },
        async submitUpload() {
            this.$refs.uploadFormRef.validate(async (valid) => {
                if (!valid) return;
                
                this.uploading = true;
                try {
                    const result = await api.uploadCase(this.uploadForm);
                    ElementPlus.ElMessage.success(`案例上传成功！已保存到 ${result.file}`);
                    this.resetUploadForm();
                    this.fetchPendingFiles();
                } catch (error) {
                    console.error('上传失败:', error);
                } finally {
                    this.uploading = false;
                }
            });
        },
        resetUploadForm() {
            this.uploadForm = {
                modality: 'text',
                fraud_type: '',
                risk_level: '高危',
                content: '',
                source: '',
                tags: [],
                uploader: 'admin'
            };
            this.refreshParsedPreview({});
            this.$refs.uploadFormRef?.resetFields();
        },
        refreshParsedPreview(parsed = {}) {
            const hasTags = Array.isArray(parsed.tags) ? parsed.tags.length > 0 : false;
            const sourceFilled = !!(parsed.source && String(parsed.source).trim());
            const contentFilled = !!(parsed.content && String(parsed.content).trim());
            const fraudTypeFilled = !!(parsed.fraud_type && String(parsed.fraud_type).trim());
            const modalityFilled = !!(parsed.modality && String(parsed.modality).trim());
            const riskLevelFilled = !!(parsed.risk_level && String(parsed.risk_level).trim());
            this.parsedFieldPreview = [
                { key: 'fraud_type', label: '诈骗类型', filled: fraudTypeFilled },
                { key: 'modality', label: '内容模态', filled: modalityFilled },
                { key: 'risk_level', label: '风险等级', filled: riskLevelFilled },
                { key: 'content', label: '案例内容', filled: contentFilled },
                { key: 'source', label: '来源', filled: sourceFilled },
                { key: 'tags', label: '标签', filled: hasTags }
            ];
        },
        handleFileChange(file) {
            if (!file) return;
            
            // 文件大小检查（5MB限制）
            if (file.raw && file.raw.size > 5 * 1024 * 1024) {
                ElementPlus.ElMessage.error('文件大小不能超过5MB');
                return;
            }
            
            const reader = new FileReader();
            const fileName = file.name.toLowerCase();
            
            reader.onload = (e) => {
                try {
                    const content = e.target.result;
                    
                    if (fileName.endsWith('.json')) {
                        // 解析JSON文件
                        this.parseJsonFile(content);
                    } else if (fileName.endsWith('.txt')) {
                        // 解析TXT文件
                        this.parseTxtFile(content);
                    }
                    
                    ElementPlus.ElMessage.success('文件解析成功，请补全缺失的属性');
                } catch (error) {
                    console.error('文件解析失败:', error);
                    ElementPlus.ElMessage.error('文件解析失败，请检查文件格式');
                }
            };
            
            reader.readAsText(file.raw);
        },
        parseJsonFile(content) {
            try {
                const data = JSON.parse(content);
                
                // 支持数组或单个对象
                const caseData = Array.isArray(data) ? data[0] : data;
                const parsed = {};
                
                // 自动填充识别的属性
                if (caseData.fraud_type) {
                    this.uploadForm.fraud_type = caseData.fraud_type;
                    parsed.fraud_type = caseData.fraud_type;
                }
                if (caseData.modality) {
                    this.uploadForm.modality = caseData.modality;
                    parsed.modality = caseData.modality;
                }
                if (caseData.risk_level) {
                    this.uploadForm.risk_level = caseData.risk_level;
                    parsed.risk_level = caseData.risk_level;
                }
                if (caseData.content) {
                    this.uploadForm.content = caseData.content;
                    parsed.content = caseData.content;
                }
                if (caseData.source) {
                    this.uploadForm.source = caseData.source;
                    parsed.source = caseData.source;
                }
                if (caseData.tags && Array.isArray(caseData.tags)) {
                    this.uploadForm.tags = caseData.tags;
                    parsed.tags = caseData.tags;
                }
                this.refreshParsedPreview(parsed);
                
                // 显示识别的属性提示
                const recognizedFields = [];
                if (caseData.fraud_type) recognizedFields.push('诈骗类型');
                if (caseData.modality) recognizedFields.push('内容模态');
                if (caseData.risk_level) recognizedFields.push('风险等级');
                if (caseData.content) recognizedFields.push('案例内容');
                
                if (recognizedFields.length > 0) {
                    ElementPlus.ElMessage.info(`已自动识别: ${recognizedFields.join('、')}`);
                }
                
            } catch (error) {
                throw new Error('JSON格式错误');
            }
        },
        parseTxtFile(content) {
            // 将TXT内容作为案例内容
            this.uploadForm.content = content.trim();
            const parsed = { content: this.uploadForm.content };
            
            // 尝试从内容中提取诈骗类型关键词
            const fraudTypeKeywords = {
                '刷单返利诈骗': ['刷单', '返利', '兼职', '垫付'],
                '虚假投资理财诈骗': ['投资', '理财', '高收益', '股票', '基金'],
                '冒充客服诈骗': ['客服', '退款', '赔偿', '订单'],
                '冒充公检法诈骗': ['公安', '警察', '检察院', '法院', '逮捕', '通缉'],
                '杀猪盘网恋诈骗': ['网恋', ' dating', '博彩', '投资平台'],
                '虚假贷款诈骗': ['贷款', '借贷', '下款', '保证金'],
                '冒充领导熟人诈骗': ['领导', '老板', '转账', '汇款'],
                '游戏产品虚假交易': ['游戏', '装备', '皮肤', '账号', '代练'],
                '婚恋交友诈骗': ['婚恋', '相亲', '交友', '礼金'],
                '消除不良记录诈骗': ['征信', '不良记录', '消除', '洗白']
            };
            
            for (const [type, keywords] of Object.entries(fraudTypeKeywords)) {
                if (keywords.some(kw => content.includes(kw))) {
                    this.uploadForm.fraud_type = type;
                    parsed.fraud_type = type;
                    ElementPlus.ElMessage.info(`从内容中识别到可能的诈骗类型: ${type}`);
                    break;
                }
            }
            
            // 自动设置为文本模态
            this.uploadForm.modality = 'text';
            parsed.modality = 'text';
            this.refreshParsedPreview(parsed);
        },
        async autoFillByLLM() {
            if (!this.uploadForm.content || !this.uploadForm.content.trim()) {
                ElementPlus.ElMessage.warning('请先上传文件或填写案例内容');
                return;
            }
            this.llmFilling = true;
            try {
                const suggestion = await api.suggestCaseFields(this.uploadForm.content);
                this.uploadForm = {
                    ...this.uploadForm,
                    fraud_type: suggestion.fraud_type || this.uploadForm.fraud_type,
                    modality: suggestion.modality || this.uploadForm.modality,
                    risk_level: suggestion.risk_level || this.uploadForm.risk_level,
                    source: suggestion.source || this.uploadForm.source,
                    tags: Array.isArray(suggestion.tags) ? suggestion.tags : this.uploadForm.tags
                };
                this.refreshParsedPreview(this.uploadForm);
                ElementPlus.ElMessage.success('LLM已完成字段建议，可再手动调整后提交');
            } catch (error) {
                console.error('LLM补全失败:', error);
            } finally {
                this.llmFilling = false;
            }
        },
        async viewPendingDetail(row) {
            try {
                const detail = await api.getPendingCaseDetail(row.filename);
                this.currentDetail = Array.isArray(detail) ? detail : (detail ? [detail] : []);
                this.detailDialogVisible = true;
            } catch (error) {
                console.error('获取详情失败:', error);
            }
        },
        async deletePendingFile(row) {
            try {
                await ElementPlus.ElMessageBox.confirm('确定删除此案例文件？', '提示', { type: 'warning' });
                await api.deletePendingCase(row.filename);
                ElementPlus.ElMessage.success('删除成功');
                this.fetchPendingFiles();
            } catch (error) {
                if (error !== 'cancel') {
                    console.error('删除失败:', error);
                }
            }
        },
        openEditDialog(row) {
            this.currentCase = row;
            this.editForm = {
                call_id: row.call_id,
                target_number: row.target_number,
                fraud_type: row.fraud_type && row.fraud_type !== '未分类拦截' ? row.fraud_type : '',
                custom_fraud_type: '',
                modality: 'audio',
                risk_level: '高危',
                details: row.details || '',
                content: row.details || '',
                tags: ['系统拦截', '高危']
            };
            this.editDialogVisible = true;
        },
        async rewriteEditContentByLLM() {
            const rawText = [this.editForm.content, this.editForm.details]
                .filter(Boolean)
                .join('\n')
                .trim();
            if (!rawText) {
                ElementPlus.ElMessage.warning('请先填写或保留原始内容再改写');
                return;
            }
            this.rewriteLoading = true;
            try {
                const fallback = await api.suggestCaseFields(rawText);
                const rewritten = this.buildNarrativeFromSuggestion(rawText, fallback);
                if (!rewritten) {
                    ElementPlus.ElMessage.warning('改写结果为空，请稍后重试');
                    return;
                }
                this.editForm.content = rewritten;
                this.editForm.details = rewritten;
                ElementPlus.ElMessage.success('已改写为案例叙事文本（不含分数信息）');
            } catch (error) {
                console.error('LLM改写失败:', error);
                if (error?.code === 'ECONNABORTED') {
                    const localNarrative = this.buildNarrativeFromSuggestion(rawText, {});
                    if (localNarrative) {
                        this.editForm.content = localNarrative;
                        this.editForm.details = localNarrative;
                        ElementPlus.ElMessage.warning('LLM响应超时，已使用本地规则完成改写');
                        return;
                    }
                }
                ElementPlus.ElMessage.error(error?.response?.data?.detail || 'LLM改写失败，请稍后重试');
            } finally {
                this.rewriteLoading = false;
            }
        },
        buildNarrativeFromSuggestion(rawText, suggestion = {}) {
            const clean = (txt) => String(txt || '')
                .replace(/\b\d+(\.\d+)?\s*%/g, '')
                .replace(/\b\d+(\.\d+)?\s*(分|score|评分|置信度)\b/gi, '')
                .replace(/(置信度|概率|评分|score|模型|算法|系统判定|AI判断)/gi, '')
                .replace(/\s+/g, ' ')
                .trim();
            const analysis = clean(suggestion.analysis);
            const advice = clean(suggestion.advice);
            const source = clean(rawText);
            const parts = [
                analysis,
                source ? `事件经过：${source}` : '',
                advice ? `风险提示：${advice}` : ''
            ].filter(Boolean);
            return parts.join('。').replace(/。{2,}/g, '。');
        },
        async submitEditLearn() {
            this.$refs.editFormRef.validate(async (valid) => {
                if (!valid) return;
                
                this.editLoading = true;
                try {
                    // 确定最终的诈骗类型
                    const finalFraudType = this.editForm.fraud_type === '其他' 
                        ? this.editForm.custom_fraud_type 
                        : this.editForm.fraud_type;
                    
                    // 构建学习数据
                    const learnData = {
                        modality: this.editForm.modality,
                        fraud_type: finalFraudType,
                        risk_level: this.editForm.risk_level,
                        content: this.editForm.content,
                        details: this.editForm.details,
                        source: `系统拦截-通话ID:${this.editForm.call_id}`,
                        tags: this.editForm.tags,
                        uploader: 'admin',
                        call_id: this.editForm.call_id
                    };
                    
                    // 调用新的API接口
                    await api.learnCaseWithEdit(this.editForm.call_id, learnData);
                    
                    ElementPlus.ElMessage.success('案例已编辑并提交到待处理库');
                    this.editDialogVisible = false;
                    this.cases = this.cases.filter(item => item.call_id !== this.editForm.call_id);
                    this.fetchPendingFiles();
                } catch (error) {
                    console.error('提交失败:', error);
                } finally {
                    this.editLoading = false;
                }
            });
        },
        async handleLearn(row) {
            // 改为必须编辑后才能学习
            this.openEditDialog(row);
        },
        formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },
        formatTime(time) {
            if (!time) return '-';
            return new Date(time).toLocaleString();
        }
    }
};