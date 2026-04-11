import AdminLayout from '../views/AdminLayout.js';
import Login from '../views/Login.js';
import Dashboard from '../views/Dashboard.js';
import RiskRules from '../views/RiskRules.js';
import Blacklist from '../views/Blacklist.js';
import CaseLearning from '../views/CaseLearning.js';
import UserManagement from '../views/UserManagement.js';
import FamilyManagement from '../views/FamilyManagement.js';

const routes = [
    { path: '/login', component: Login, name: '登录' },
    {
        path: '/',
        component: AdminLayout,
        children: [
            { path: '', component: Dashboard, name: '监控仪表盘' },
            { path: 'rules', component: RiskRules, name: '风控规则库' },
            { path: 'blacklist', component: Blacklist, name: '黑名单数据库' },
            { path: 'case-learning', component: CaseLearning, name: '案例学习' },
            { path: 'users', component: UserManagement, name: '用户管理' },
            { path: 'families', component: FamilyManagement, name: '家庭组管理' }
        ]
    }
];

const router = VueRouter.createRouter({
    history: VueRouter.createWebHashHistory(),
    routes
});

router.beforeEach((to, from, next) => {
    const token = localStorage.getItem('admin_token');
    if (to.path === '/login') {
        if (token) return next('/');
        return next();
    }
    if (!token) return next('/login');
    next();
});

const app = Vue.createApp({});
app.use(router);
app.use(ElementPlus);
app.mount('#app');
