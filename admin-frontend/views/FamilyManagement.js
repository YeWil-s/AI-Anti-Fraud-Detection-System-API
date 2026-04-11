import api from '../js/api.js';

export default {
    template: `
        <div class="page-card">
            <div class="page-header">
                <div>
                    <div class="page-title">家庭组管理</div>
                    <div style="color: #6b7280; font-size: 13px; margin-top: 5px;">
                        管理家庭组、成员、管理员权限
                    </div>
                </div>
                <el-button type="primary" size="large" @click="showCreateDialog = true">
                    <i class="ri-add-line"></i> 创建家庭组
                </el-button>
            </div>

            <!-- 统计卡片 -->
            <el-row :gutter="20" style="margin-bottom: 20px;">
                <el-col :span="8">
                    <div class="stat-card stat-card-gradient-blue">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ stats.total_families || 0 }}</div>
                            <div class="label">家庭组总数</div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="8">
                    <div class="stat-card stat-card-gradient-green">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ stats.total_members || 0 }}</div>
                            <div class="label">家庭成员总数</div>
                        </div>
                    </div>
                </el-col>
                <el-col :span="8">
                    <div class="stat-card stat-card-gradient-purple">
                        <div class="stat-info" style="color: white;">
                            <div class="num">{{ stats.total_admins || 0 }}</div>
                            <div class="label">管理员总数</div>
                        </div>
                    </div>
                </el-col>
            </el-row>

            <!-- 搜索栏 -->
            <div style="margin-bottom: 20px; display: flex; gap: 10px;">
                <el-input 
                    v-model="searchQuery" 
                    placeholder="搜索家庭组名称" 
                    style="width: 300px;"
                    clearable
                    @input="handleSearch">
                    <template #prefix>
                        <i class="ri-search-line"></i>
                    </template>
                </el-input>
            </div>

            <!-- 家庭组表格 -->
            <el-table :data="filteredFamilies" v-loading="loading" stripe border>
                <el-table-column type="index" label="#" width="60" align="center"></el-table-column>
                <el-table-column prop="id" label="ID" width="80" align="center"></el-table-column>
                <el-table-column prop="group_name" label="家庭组名称" min-width="150">
                    <template #default="scope">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <i class="ri-home-heart-line" style="font-size: 18px; color: #4f46e5;"></i>
                            <span>{{ scope.row.group_name }}</span>
                        </div>
                    </template>
                </el-table-column>
                <el-table-column label="主管理员" width="150">
                    <template #default="scope">
                        <span v-if="scope.row.primary_admin">
                            {{ scope.row.primary_admin.username }}
                        </span>
                        <span v-else style="color: #999;">-</span>
                    </template>
                </el-table-column>
                <el-table-column label="成员数" width="100" align="center">
                    <template #default="scope">
                        <el-tag type="info" size="small">
                            {{ scope.row.statistics?.total_members || 0 }} 人
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column label="管理员" width="150" align="center">
                    <template #default="scope">
                        <div>
                            <el-tag type="danger" size="small" v-if="scope.row.statistics?.primary_admins">
                                主{{ scope.row.statistics.primary_admins }}
                            </el-tag>
                            <el-tag type="warning" size="small" v-if="scope.row.statistics?.secondary_admins" style="margin-left: 5px;">
                                副{{ scope.row.statistics.secondary_admins }}
                            </el-tag>
                        </div>
                    </template>
                </el-table-column>
                <el-table-column prop="created_at" label="创建时间" width="160">
                    <template #default="scope">
                        {{ formatTime(scope.row.created_at) }}
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="200" align="center" fixed="right">
                    <template #default="scope">
                        <el-button link type="primary" @click="viewFamilyDetail(scope.row)">详情</el-button>
                        <el-button link type="primary" @click="manageMembers(scope.row)">成员</el-button>
                        <el-button link type="danger" @click="deleteFamily(scope.row)">删除</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <el-empty v-if="!loading && filteredFamilies.length === 0" description="暂无家庭组"></el-empty>

            <!-- 创建家庭组对话框 -->
            <el-dialog v-model="showCreateDialog" title="创建家庭组" width="500px">
                <el-form :model="createForm" label-width="100px">
                    <el-form-item label="家庭组名称" required>
                        <el-input v-model="createForm.name" placeholder="请输入家庭组名称" />
                    </el-form-item>
                </el-form>
                <template #footer>
                    <el-button @click="showCreateDialog = false">取消</el-button>
                    <el-button type="primary" @click="createFamily" :loading="creating">创建</el-button>
                </template>
            </el-dialog>

            <!-- 家庭组详情对话框 -->
            <el-dialog v-model="detailVisible" title="家庭组详情" width="700px">
                <div v-if="currentFamily">
                    <el-descriptions :column="2" border>
                        <el-descriptions-item label="家庭组ID">{{ currentFamily.id }}</el-descriptions-item>
                        <el-descriptions-item label="家庭组名称">{{ currentFamily.group_name }}</el-descriptions-item>
                        <el-descriptions-item label="主管理员">
                            {{ currentFamily.primary_admin?.username || '-' }}
                            ({{ currentFamily.primary_admin?.phone || '-' }})
                        </el-descriptions-item>
                        <el-descriptions-item label="创建时间">
                            {{ formatTime(currentFamily.created_at) }}
                        </el-descriptions-item>
                    </el-descriptions>

                    <!-- 统计信息 -->
                    <div style="margin-top: 20px;">
                        <div style="font-weight: 600; margin-bottom: 15px;">成员统计</div>
                        <el-row :gutter="20">
                            <el-col :span="8">
                                <div style="background: #f0f9ff; padding: 15px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold; color: #0369a1;">
                                        {{ currentFamily.statistics?.total_members || 0 }}
                                    </div>
                                    <div style="font-size: 12px; color: #64748b;">总成员</div>
                                </div>
                            </el-col>
                            <el-col :span="8">
                                <div style="background: #fef2f2; padding: 15px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold; color: #dc2626;">
                                        {{ currentFamily.statistics?.primary_admins || 0 }}
                                    </div>
                                    <div style="font-size: 12px; color: #64748b;">主管理员</div>
                                </div>
                            </el-col>
                            <el-col :span="8">
                                <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 24px; font-weight: bold; color: #16a34a;">
                                        {{ currentFamily.statistics?.secondary_admins || 0 }}
                                    </div>
                                    <div style="font-size: 12px; color: #64748b;">副管理员</div>
                                </div>
                            </el-col>
                        </el-row>
                    </div>

                    <!-- 成员列表 -->
                    <div style="margin-top: 20px;">
                        <div style="font-weight: 600; margin-bottom: 15px;">成员列表</div>
                        <el-table :data="currentFamily.members" border size="small">
                            <el-table-column prop="username" label="用户名" width="120"></el-table-column>
                            <el-table-column prop="name" label="姓名" width="100"></el-table-column>
                            <el-table-column prop="phone" label="手机号" width="130"></el-table-column>
                            <el-table-column prop="role_type" label="角色" width="100">
                                <template #default="scope">
                                    <el-tag :type="getRoleType(scope.row.role_type)" size="small">
                                        {{ scope.row.role_type || '未设置' }}
                                    </el-tag>
                                </template>
                            </el-table-column>
                            <el-table-column prop="admin_role" label="管理权限" width="100">
                                <template #default="scope">
                                    <el-tag v-if="scope.row.admin_role === 'primary'" type="danger" size="small">主管理员</el-tag>
                                    <el-tag v-else-if="scope.row.admin_role === 'secondary'" type="warning" size="small">副管理员</el-tag>
                                    <span v-else style="color: #999;">普通成员</span>
                                </template>
                            </el-table-column>
                        </el-table>
                    </div>
                </div>
            </el-dialog>

            <!-- 成员管理对话框 -->
            <el-dialog v-model="membersVisible" title="成员管理" width="800px">
                <div v-if="currentFamily">
                    <div style="margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                        <span>家庭组: {{ currentFamily.group_name }}</span>
                        <el-button type="primary" size="small" @click="showAddMemberDialog = true">
                            <i class="ri-user-add-line"></i> 添加成员
                        </el-button>
                    </div>
                    
                    <el-table :data="currentFamily.members" border>
                        <el-table-column prop="username" label="用户名" width="120"></el-table-column>
                        <el-table-column prop="name" label="姓名" width="100"></el-table-column>
                        <el-table-column prop="phone" label="手机号" width="130"></el-table-column>
                        <el-table-column prop="role_type" label="角色" width="100">
                            <template #default="scope">
                                <el-tag :type="getRoleType(scope.row.role_type)" size="small">
                                    {{ scope.row.role_type || '未设置' }}
                                </el-tag>
                            </template>
                        </el-table-column>
                        <el-table-column prop="admin_role" label="管理权限" width="120">
                            <template #default="scope">
                                <el-select 
                                    v-model="scope.row.admin_role" 
                                    size="small" 
                                    style="width: 100px;"
                                    @change="(val) => updateAdminRole(scope.row, val)">
                                    <el-option label="普通成员" value="none"></el-option>
                                    <el-option label="副管理员" value="secondary"></el-option>
                                    <el-option label="主管理员" value="primary"></el-option>
                                </el-select>
                            </template>
                        </el-table-column>
                        <el-table-column label="操作" width="100" align="center">
                            <template #default="scope">
                                <el-button link type="danger" @click="removeMember(scope.row)">移除</el-button>
                            </template>
                        </el-table-column>
                    </el-table>
                </div>
            </el-dialog>

            <!-- 添加成员对话框 -->
            <el-dialog v-model="showAddMemberDialog" title="添加成员" width="450px">
                <el-form :model="addMemberForm" label-width="80px">
                    <el-form-item label="用户ID">
                        <el-input v-model="addMemberForm.userId" placeholder="请输入要添加的用户ID" type="number" />
                    </el-form-item>
                    <el-alert
                        title="提示：请输入已注册用户的ID，用户将被添加到当前家庭组"
                        type="info"
                        :closable="false"
                        style="margin-top: 10px;"
                    />
                </el-form>
                <template #footer>
                    <el-button @click="showAddMemberDialog = false">取消</el-button>
                    <el-button type="primary" @click="doAddMember" :loading="addingMember">确认添加</el-button>
                </template>
            </el-dialog>
        </div>
    `,
    data() {
        return {
            loading: false,
            creating: false,
            families: [],
            filteredFamilies: [],
            stats: {},
            searchQuery: '',
            showCreateDialog: false,
            detailVisible: false,
            membersVisible: false,
            showAddMemberDialog: false,
            addingMember: false,
            currentFamily: null,
            createForm: { name: '' },
            addMemberForm: { userId: '' }
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
                    this.loadFamilies(),
                    this.loadStats()
                ]);
            } finally {
                this.loading = false;
            }
        },
        async loadFamilies() {
            try {
                const res = await api.getFamilyGroups();
                this.families = res.items || [];
                this.filteredFamilies = this.families;
            } catch (error) {
                console.error('加载家庭组失败:', error);
                this.families = [];
                this.filteredFamilies = [];
            }
        },
        async loadStats() {
            try {
                const res = await api.getFamilyStats?.() || {};
                this.stats = res || {};
            } catch (error) {
                this.stats = {};
            }
        },
        handleSearch() {
            if (!this.searchQuery) {
                this.filteredFamilies = this.families;
                return;
            }
            this.filteredFamilies = this.families.filter(f => 
                f.group_name?.toLowerCase().includes(this.searchQuery.toLowerCase())
            );
        },
        async createFamily() {
            if (!this.createForm.name.trim()) {
                ElementPlus.ElMessage.warning('请输入家庭组名称');
                return;
            }
            this.creating = true;
            try {
                await api.createFamilyGroup(this.createForm.name);
                ElementPlus.ElMessage.success('家庭组创建成功');
                this.showCreateDialog = false;
                this.createForm.name = '';
                this.loadFamilies();
            } catch (error) {
                console.error('创建失败:', error);
                ElementPlus.ElMessage.error('创建失败');
            } finally {
                this.creating = false;
            }
        },
        async viewFamilyDetail(row) {
            try {
                const res = await api.getFamilyGroupMembers(row.id);
                this.currentFamily = {
                    ...row,
                    members: res.members || []
                };
                this.detailVisible = true;
            } catch (error) {
                console.error('加载详情失败:', error);
            }
        },
        async manageMembers(row) {
            try {
                const res = await api.getFamilyGroupMembers(row.id);
                this.currentFamily = {
                    ...row,
                    members: res.members || []
                };
                this.membersVisible = true;
            } catch (error) {
                console.error('加载成员失败:', error);
            }
        },
        async updateAdminRole(member, newRole) {
            try {
                await api.setMemberAdminRole(member.user_id, newRole);
                ElementPlus.ElMessage.success('权限更新成功');
                // 刷新当前家庭组数据
                if (this.currentFamily) {
                    this.manageMembers(this.currentFamily);
                }
            } catch (error) {
                console.error('更新权限失败:', error);
                ElementPlus.ElMessage.error('更新失败');
            }
        },
        async removeMember(member) {
            try {
                await ElementPlus.ElMessageBox.confirm(
                    `确定将 ${member.username} 移出家庭组？`, 
                    '确认移除', 
                    { type: 'warning' }
                );
                await api.removeFamilyMember(member.user_id);
                ElementPlus.ElMessage.success('移除成功');
                // 刷新列表
                if (this.currentFamily) {
                    this.manageMembers(this.currentFamily);
                }
            } catch (error) {
                if (error !== 'cancel') {
                    console.error('移除失败:', error);
                }
            }
        },
        async deleteFamily(family) {
            try {
                await ElementPlus.ElMessageBox.confirm(
                    `确定删除家庭组 "${family.group_name}"？此操作不可恢复！`, 
                    '警告', 
                    { type: 'warning' }
                );
                await api.deleteFamilyGroup(family.id);
                ElementPlus.ElMessage.success('删除成功');
                this.loadFamilies();
            } catch (error) {
                if (error !== 'cancel') {
                    console.error('删除失败:', error);
                    ElementPlus.ElMessage.error('删除失败');
                }
            }
        },
        async doAddMember() {
            if (!this.addMemberForm.userId) {
                ElementPlus.ElMessage.warning('请输入用户ID');
                return;
            }
            this.addingMember = true;
            try {
                await api.addFamilyMember(this.currentFamily.id, this.addMemberForm.userId);
                ElementPlus.ElMessage.success('成员添加成功');
                this.showAddMemberDialog = false;
                this.addMemberForm.userId = '';
                // 刷新成员列表
                this.manageMembers(this.currentFamily);
            } catch (error) {
                console.error('添加成员失败:', error);
                ElementPlus.ElMessage.error('添加失败，请检查用户ID是否正确');
            } finally {
                this.addingMember = false;
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
