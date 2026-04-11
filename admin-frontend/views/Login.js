import api from '../js/api.js';

export default {
    data() {
        return {
            tab: 'login',
            username: '',
            password: '',
            regUsername: '',
            regPassword: '',
            regPassword2: '',
            registerSecret: '',
            loading: false
        };
    },
    methods: {
        async onLogin() {
            const u = (this.username || '').trim();
            if (!u) {
                ElementPlus.ElMessage.warning('请输入用户名');
                return;
            }
            if (!this.password) {
                ElementPlus.ElMessage.warning('请输入密码');
                return;
            }
            this.loading = true;
            try {
                const data = await api.adminLogin(u, this.password);
                if (data && data.access_token) {
                    localStorage.setItem('admin_token', data.access_token);
                    await this.$router.replace('/');
                }
            } catch (_) {
                /* 错误信息由 api 拦截器统一提示 */
            } finally {
                this.loading = false;
            }
        },
        async onRegister() {
            const u = (this.regUsername || '').trim();
            if (!u) {
                ElementPlus.ElMessage.warning('请输入用户名');
                return;
            }
            if (u.length < 2) {
                ElementPlus.ElMessage.warning('用户名至少 2 个字符');
                return;
            }
            if (!this.regPassword || this.regPassword.length < 6) {
                ElementPlus.ElMessage.warning('密码至少 6 位');
                return;
            }
            if (this.regPassword !== this.regPassword2) {
                ElementPlus.ElMessage.warning('两次输入的密码不一致');
                return;
            }
            if (!(this.registerSecret || '').trim()) {
                ElementPlus.ElMessage.warning('请输入管理员密钥');
                return;
            }
            this.loading = true;
            try {
                await api.adminRegister({
                    username: u,
                    password: this.regPassword,
                    register_secret: this.registerSecret.trim()
                });
                ElementPlus.ElMessage.success('注册成功，请登录');
                this.username = u;
                this.password = '';
                this.regPassword = '';
                this.regPassword2 = '';
                this.registerSecret = '';
                this.tab = 'login';
            } catch (_) {
                /* 错误信息由 api 拦截器统一提示 */
            } finally {
                this.loading = false;
            }
        }
    },
    template: `
        <div class="admin-login-wrap">
            <div class="admin-login-card">
                <div class="admin-login-title"><i class="ri-shield-keyhole-line"></i> 管理后台</div>
                <p class="admin-login-sub">使用 admins 账号登录，或通过密钥注册新管理员</p>
                <el-tabs v-model="tab" stretch class="admin-login-tabs">
                    <el-tab-pane label="登录" name="login">
                        <el-form @submit.prevent="onLogin" label-position="top" style="margin-top:8px">
                            <el-form-item label="用户名">
                                <el-input v-model="username" autocomplete="username" placeholder="管理员用户名" clearable @keyup.enter="onLogin" />
                            </el-form-item>
                            <el-form-item label="密码">
                                <el-input v-model="password" type="password" autocomplete="current-password" placeholder="密码" show-password @keyup.enter="onLogin" />
                            </el-form-item>
                            <el-button type="primary" style="width:100%" :loading="loading" @click="onLogin">登录</el-button>
                        </el-form>
                    </el-tab-pane>
                    <el-tab-pane label="注册" name="register">
                        <el-form @submit.prevent="onRegister" label-position="top" style="margin-top:8px">
                            <el-form-item label="用户名">
                                <el-input v-model="regUsername" autocomplete="off" placeholder="至少 2 个字符" clearable />
                            </el-form-item>
                            <el-form-item label="密码">
                                <el-input v-model="regPassword" type="password" autocomplete="new-password" placeholder="至少 6 位" show-password />
                            </el-form-item>
                            <el-form-item label="确认密码">
                                <el-input v-model="regPassword2" type="password" autocomplete="new-password" placeholder="再次输入密码" show-password @keyup.enter="onRegister" />
                            </el-form-item>
                            <el-form-item label="管理员密钥">
                                <el-input v-model="registerSecret" type="password" show-password placeholder="注册必填，请联系部署方获取" autocomplete="off" />
                            </el-form-item>
                            <el-button type="primary" style="width:100%" :loading="loading" @click="onRegister">注册管理员</el-button>
                        </el-form>
                    </el-tab-pane>
                </el-tabs>
            </div>
        </div>
    `
};
