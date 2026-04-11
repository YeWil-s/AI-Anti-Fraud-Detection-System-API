export default {
    computed: {
        activePath() {
            return this.$route.path;
        },
        currentRouteName() {
            return this.$route.name || '';
        }
    },
    methods: {
        logout() {
            localStorage.removeItem('admin_token');
            this.$router.replace('/login');
        }
    },
    template: `
        <el-container class="app-wrapper">
            <el-aside width="220px" class="sidebar">
                <div class="logo"><i class="ri-shield-check-fill"></i> AI Sentinel</div>
                <el-menu :default-active="activePath" router background-color="#1f2937" text-color="#9ca3af" active-text-color="#fff">
                    <el-menu-item index="/">
                        <i class="ri-dashboard-line"></i>
                        <span style="margin-left:8px">监控仪表盘</span>
                    </el-menu-item>
                    <el-menu-item index="/rules">
                        <i class="ri-list-settings-line"></i>
                        <span style="margin-left:8px">关键词规则</span>
                    </el-menu-item>
                    <el-menu-item index="/blacklist">
                        <i class="ri-spam-line"></i>
                        <span style="margin-left:8px">黑名单管理</span>
                    </el-menu-item>
                    <el-menu-item index="/case-learning">
                        <i class="ri-brain-line"></i>
                        <span style="margin-left:8px">AI知识库</span>
                    </el-menu-item>
                    <el-menu-item index="/users">
                        <i class="ri-user-settings-line"></i>
                        <span style="margin-left:8px">用户管理</span>
                    </el-menu-item>
                    <el-menu-item index="/families">
                        <i class="ri-home-heart-line"></i>
                        <span style="margin-left:8px">家庭组管理</span>
                    </el-menu-item>
                </el-menu>
            </el-aside>
            <el-container>
                <el-header class="header" style="display:flex;align-items:center;justify-content:space-between;">
                    <div class="breadcrumb">管理控制台 / {{ currentRouteName }}</div>
                    <div style="display:flex;align-items:center;gap:12px;">
                        <el-tag type="success" size="small">API 连接状态: 在线</el-tag>
                        <el-button size="small" @click="logout">退出登录</el-button>
                    </div>
                </el-header>
                <el-main>
                    <router-view></router-view>
                </el-main>
            </el-container>
        </el-container>
    `
};
