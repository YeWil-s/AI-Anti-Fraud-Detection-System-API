import api from '../js/api.js';

export default {
    template: `
        <div class="page-card">
            <div class="page-header">
                <div>
                    <div class="page-title">用户管理</div>
                    <div style="color: #6b7280; font-size: 13px; margin-top: 5px;">
                        管理系统用户、查看用户画像、管理家庭组
                    </div>
                </div>
                <el-button type="primary" size="large" @click="showAddDialog = true">
                    <i class="ri-user-add-line"></i> 添加用户
                </el-button>
            </div>

            <!-- 统计卡片 -->
            <el-row :gutter="20" style="margin-bottom: 20px;">
                <el-col :span="6">
                    <div class="stat-card stat-card-gradient-blue">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ stats.total_users || 0 }}</div>
                            <div class="label">总用户数</div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card stat-card-gradient-green">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ stats.active_users || 0 }}</div>
                            <div class="label">活跃用户</div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card stat-card-gradient-purple">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ stats.new_users_today || 0 }}</div>
                            <div class="label">今日新增</div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="6">
                    <div class="stat-card stat-card-gradient-pink">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ familyGroups.length || 0 }}</div>
                            <div class="label">家庭组数</div>
                        </div>
                    </div>
                </el-col>
            </el-row>

            <!-- 搜索栏 -->
            <div style="margin-bottom: 20px; display: flex; gap: 10px;">
                <el-input 
                    v-model="searchQuery" 
                    placeholder="搜索用户名、手机号或邮箱" 
                    style="width: 300px;"
                    clearable
                    @input="handleSearch">
                    <template #prefix>
                        <i class="ri-search-line"></i>
                    </template>
                </el-input>
                <el-select v-model="filterRole" placeholder="角色类型" clearable @change="handleSearch" style="width: 150px;">
                    <el-option label="老人" value="老人"></el-option>
                    <el-option label="儿童" value="儿童"></el-option>
                    <el-option label="学生" value="学生"></el-option>
                    <el-option label="青壮年" value="青壮年"></el-option>
                </el-select>
                <el-select v-model="filterStatus" placeholder="账号状态" clearable @change="handleSearch" style="width: 150px;">
                    <el-option label="正常" :value="true"></el-option>
                    <el-option label="禁用" :value="false"></el-option>
                </el-select>
            </div>

            <!-- 用户表格 -->
            <el-table :data="filteredUsers" v-loading="loading" stripe border>
                <el-table-column type="index" label="#" width="60" align="center"></el-table-column>
                <el-table-column prop="user_id" label="ID" width="80" align="center"></el-table-column>
                <el-table-column prop="username" label="用户名" width="150">
                    <template #default="scope">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <i class="ri-user-line" style="font-size: 18px; color: #4f46e5;"></i>
                            <span>{{ scope.row.username }}</span>
                        </div>
                    </template>
                </el-table-column>
                <el-table-column prop="name" label="姓名" width="100"></el-table-column>
                <el-table-column prop="phone" label="手机号" width="130"></el-table-column>
                <el-table-column prop="email" label="邮箱" min-width="180"></el-table-column>
                <el-table-column prop="role_type" label="角色" width="100" align="center">
                    <template #default="scope">
                        <el-tag :type="getRoleType(scope.row.role_type)" size="small">
                            {{ scope.row.role_type || '未设置' }}
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="family_id" label="家庭组" width="100" align="center">
                    <template #default="scope">
                        <el-tag v-if="scope.row.family_id" type="success" size="small">已加入</el-tag>
                        <span v-else style="color: #999;">-</span>
                    </template>
                </el-table-column>
                <el-table-column prop="is_active" label="状态" width="80" align="center">
                    <template #default="scope">
                        <el-switch 
                            v-model="scope.row.is_active" 
                            @change="toggleUserStatus(scope.row)"
                            inline-prompt
                            active-text="正常"
                            inactive-text="禁用"
                        />
                    </template>
                </el-table-column>
                <el-table-column prop="created_at" label="注册时间" width="160">
                    <template #default="scope">
                        {{ formatTime(scope.row.created_at) }}
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="200" align="center" fixed="right">
                    <template #default="scope">
                        <el-button link type="primary" @click="viewUserDetail(scope.row)">详情</el-button>
                        <el-button link type="primary" @click="editUser(scope.row)">编辑</el-button>
                        <el-button link type="danger" @click="deleteUser(scope.row)">删除</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <el-empty v-if="!loading && filteredUsers.length === 0" description="暂无匹配的用户"></el-empty>

            <!-- 用户详情对话框 -->
            <el-dialog v-model="detailVisible" title="用户详情" width="700px">
                <el-descriptions :column="2" border v-if="currentUser">
                    <el-descriptions-item label="用户ID">{{ currentUser.user_id }}</el-descriptions-item>
                    <el-descriptions-item label="用户名">{{ currentUser.username }}</el-descriptions-item>
                    <el-descriptions-item label="姓名">{{ currentUser.name || '-' }}</el-descriptions-item>
                    <el-descriptions-item label="性别">{{ currentUser.gender || '-' }}</el-descriptions-item>
                    <el-descriptions-item label="手机号">{{ currentUser.phone }}</el-descriptions-item>
                    <el-descriptions-item label="邮箱">{{ currentUser.email || '-' }}</el-descriptions-item>
                    <el-descriptions-item label="角色类型">
                        <el-tag :type="getRoleType(currentUser.role_type)">
                            {{ currentUser.role_type || '未设置' }}
                        </el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="职业">{{ currentUser.profession || '-' }}</el-descriptions-item>
                    <el-descriptions-item label="婚姻状况">{{ currentUser.marital_status || '-' }}</el-descriptions-item>
                    <el-descriptions-item label="家庭组ID">{{ currentUser.family_id || '-' }}</el-descriptions-item>
                    <el-descriptions-item label="账号状态">
                        <el-tag :type="currentUser.is_active ? 'success' : 'danger'">
                            {{ currentUser.is_active ? '正常' : '禁用' }}
                        </el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="是否管理员">{{ currentUser.is_admin ? '是' : '否' }}</el-descriptions-item>
                    <el-descriptions-item label="注册时间" :span="2">
                        {{ formatTime(currentUser.created_at) }}
                    </el-descriptions-item>
                </el-descriptions>

                <!-- 用户通话统计 -->
                <div style="margin-top: 20px;">
                    <div style="font-weight: 600; margin-bottom: 15px;">通话统计</div>
                    <el-row :gutter="20">
                        <el-col :span="8">
                            <div style="background: #f0f9ff; padding: 15px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 24px; font-weight: bold; color: #0369a1;">
                                    {{ userStats.total_calls || 0 }}
                                </div>
                                <div style="font-size: 12px; color: #64748b;">总通话数</div>
                            </div>
                        </el-col>
                        <el-col :span="8">
                            <div style="background: #fef2f2; padding: 15px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 24px; font-weight: bold; color: #dc2626;">
                                    {{ userStats.fraud_calls || 0 }}
                                </div>
                                <div style="font-size: 12px; color: #64748b;">诈骗拦截</div>
                            </div>
                        </el-col>
                        <el-col :span="8">
                            <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 24px; font-weight: bold; color: #16a34a;">
                                    {{ userStats.suspicious_calls || 0 }}
                                </div>
                                <div style="font-size: 12px; color: #64748b;">可疑通话</div>
                            </div>
                        </el-col>
                    </el-row>
                </div>
            </el-dialog>

            <!-- 编辑用户对话框 -->
            <el-dialog v-model="editVisible" title="编辑用户" width="550px">
                <el-form :model="editForm" label-width="100px" v-if="editForm">
                    <el-form-item label="用户名">
                        <el-input v-model="editForm.username" disabled />
                    </el-form-item>
                    <el-form-item label="姓名">
                        <el-input v-model="editForm.name" />
                    </el-form-item>
                    <el-form-item label="手机号">
                        <el-input v-model="editForm.phone" />
                    </el-form-item>
                    <el-form-item label="邮箱">
                        <el-input v-model="editForm.email" />
                    </el-form-item>
                    <el-form-item label="角色类型">
                        <el-select v-model="editForm.role_type" style="width: 100%;">
                            <el-option label="老人" value="老人"></el-option>
                            <el-option label="儿童" value="儿童"></el-option>
                            <el-option label="学生" value="学生"></el-option>
                            <el-option label="青壮年" value="青壮年"></el-option>
                        </el-select>
                    </el-form-item>
                    <el-form-item label="性别">
                        <el-radio-group v-model="editForm.gender">
                            <el-radio label="男">男</el-radio>
                            <el-radio label="女">女</el-radio>
                            <el-radio label="未知">未知</el-radio>
                        </el-radio-group>
                    </el-form-item>
                    <el-form-item label="职业">
                        <el-input v-model="editForm.profession" />
                    </el-form-item>
                    <el-form-item label="婚姻状况">
                        <el-select v-model="editForm.marital_status" style="width: 100%;">
                            <el-option label="单身" value="单身"></el-option>
                            <el-option label="已婚" value="已婚"></el-option>
                            <el-option label="离异" value="离异"></el-option>
                        </el-select>
                    </el-form-item>
                </el-form>
                <template #footer>
                    <el-button @click="editVisible = false">取消</el-button>
                    <el-button type="primary" @click="saveUser" :loading="saving">保存</el-button>
                </template>
            </el-dialog>
        </div>
    `,
    data() {
        return {
            loading: false,
            saving: false,
            users: [],
            filteredUsers: [],
            familyGroups: [],
            stats: {},
            searchQuery: '',
            filterRole: '',
            filterStatus: '',
            detailVisible: false,
            editVisible: false,
            showAddDialog: false,
            currentUser: null,
            userStats: {},
            editForm: null
        };
    },
    mounted() {
        this.loadData();
    },
    methods: {
        async loadData() {
            this.loading = true;
            try {
                await Promise.all([
                    this.loadUsers(),
                    this.loadStats(),
                    this.loadFamilyGroups()
                ]);
            } finally {
                this.loading = false;
            }
        },
        async loadUsers() {
            try {
                this.users = await api.getUsers();
                this.filteredUsers = this.users;
            } catch (error) {
                console.error('加载用户失败:', error);
                // 使用模拟数据
                this.users = [
                    { user_id: 1, username: 'user1', name: '张三', phone: '13800138001', email: 'user1@test.com', role_type: '青壮年', is_active: true, created_at: new Date().toISOString() },
                    { user_id: 2, username: 'user2', name: '李四', phone: '13800138002', email: 'user2@test.com', role_type: '老人', is_active: true, created_at: new Date().toISOString() }
                ];
                this.filteredUsers = this.users;
            }
        },
        async loadStats() {
            try {
                this.stats = await api.getStats();
            } catch (error) {
                this.stats = { total_users: 0, active_users: 0, new_users_today: 0 };
            }
        },
        async loadFamilyGroups() {
            try {
                this.familyGroups = await api.getFamilyGroups();
            } catch (error) {
                this.familyGroups = [];
            }
        },
        handleSearch() {
            this.filteredUsers = this.users.filter(user => {
                const matchQuery = !this.searchQuery || 
                    user.username?.toLowerCase().includes(this.searchQuery.toLowerCase()) ||
                    user.phone?.includes(this.searchQuery) ||
                    user.email?.toLowerCase().includes(this.searchQuery.toLowerCase());
                const matchRole = !this.filterRole || user.role_type === this.filterRole;
                const matchStatus = this.filterStatus === '' || user.is_active === this.filterStatus;
                return matchQuery && matchRole && matchStatus;
            });
        },
        async viewUserDetail(row) {
            this.currentUser = row;
            // 加载用户通话统计
            try {
                this.userStats = await api.getUserCallStats(row.user_id);
            } catch (error) {
                this.userStats = { total_calls: 0, fraud_calls: 0, suspicious_calls: 0 };
            }
            this.detailVisible = true;
        },
        editUser(row) {
            this.editForm = { ...row };
            this.editVisible = true;
        },
        async saveUser() {
            this.saving = true;
            try {
                await api.updateUser(this.editForm.user_id, this.editForm);
                ElementPlus.ElMessage.success('保存成功');
                this.editVisible = false;
                this.loadUsers();
            } catch (error) {
                console.error('保存失败:', error);
            } finally {
                this.saving = false;
            }
        },
        async toggleUserStatus(row) {
            try {
                await api.updateUserStatus(row.user_id, row.is_active);
                ElementPlus.ElMessage.success(row.is_active ? '账号已启用' : '账号已禁用');
            } catch (error) {
                row.is_active = !row.is_active; // 回滚
                console.error('切换状态失败:', error);
            }
        },
        async deleteUser(row) {
            try {
                await ElementPlus.ElMessageBox.confirm('确定删除该用户？此操作不可恢复！', '警告', { type: 'warning' });
                await api.deleteUser(row.user_id);
                ElementPlus.ElMessage.success('删除成功');
                this.loadUsers();
            } catch (error) {
                if (error !== 'cancel') {
                    console.error('删除失败:', error);
                }
            }
        },
        getRoleType(role) {
            const map = { '老人': 'danger', '儿童': 'warning', '学生': 'success', '青壮年': 'primary' };
            return map[role] || 'info';
        },
        formatTime(time) {
            if (!time) return '-';
            return new Date(time).toLocaleString();
        }
    }
};
