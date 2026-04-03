import api from './api.js';
import Dashboard from '../views/Dashboard.js';
import RiskRules from '../views/RiskRules.js';
import Blacklist from '../views/Blacklist.js';
import TestConsole from '../views/TestConsole.js';
import CaseLearning from '../views/CaseLearning.js';
import UserManagement from '../views/UserManagement.js';
import FamilyManagement from '../views/FamilyManagement.js';

const routes = [
    { path: '/', component: Dashboard, name: '监控仪表盘' },
    { path: '/rules', component: RiskRules, name: '风控规则库' },
    { path: '/blacklist', component: Blacklist, name: '黑名单数据库' },
    { path: '/test', component: TestConsole, name: '功能测试台' },
    { path: '/case-learning', component: CaseLearning, name: '案例学习' },
    { path: '/users', component: UserManagement, name: '用户管理' },
    { path: '/families', component: FamilyManagement, name: '家庭组管理' }
];

const router = VueRouter.createRouter({
    history: VueRouter.createWebHashHistory(),
    routes,
});

// 路由权限守卫
router.beforeEach((to, from, next) => {
    const token = localStorage.getItem('admin_token');
    // 如果访问的不是登录页，且没有 token，则可以提示或跳转
    // 由于是管理后台，简单检查 token 存在性即可
    if (!token && to.path !== '/login') {
        // 如果没有登录页路由，则只记录警告，不阻止导航
        console.warn('未检测到管理员认证信息');
    }
    next();
});

const app = Vue.createApp({
    data() {
        return { activePath: '/', currentRouteName: '仪表盘' }
    },
    created() {
        this.$router.afterEach((to) => {
            this.activePath = to.path;
            this.currentRouteName = to.name;
        });
    }
});

app.use(router);
app.use(ElementPlus);
app.mount('#app');