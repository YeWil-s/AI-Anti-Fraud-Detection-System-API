import api from './api.js';
import Dashboard from '../views/Dashboard.js';
import RiskRules from '../views/RiskRules.js';
import Blacklist from '../views/Blacklist.js';
import TestConsole from '../views/TestConsole.js';

const routes = [
    { path: '/', component: Dashboard, name: '监控仪表盘' },
    { path: '/rules', component: RiskRules, name: '风控规则库' },
    { path: '/blacklist', component: Blacklist, name: '黑名单数据库' },
    { path: '/test', component: TestConsole, name: '功能测试台' },
];

const router = VueRouter.createRouter({
    history: VueRouter.createWebHashHistory(),
    routes,
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