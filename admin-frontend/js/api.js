const api = axios.create({
    baseURL: 'http://localhost:8000/api',
    timeout: 10000
});

// 响应拦截
api.interceptors.response.use(
    res => res.data,
    err => {
        ElementPlus.ElMessage.error(err.response?.data?.detail || '请求失败，请检查后端服务');
        return Promise.reject(err);
    }
);

export default {
    // 仪表盘
    getStats: () => api.get('/admin/stats'),
    
    // 规则管理
    getRules: () => api.get('/admin/rules'),
    addRule: (data) => api.post('/admin/rules', data),
    delRule: (id) => api.delete(`/admin/rules/${id}`),

    // 黑名单管理
    getBlacklist: () => api.get('/admin/blacklist'),
    addBlacklist: (data) => api.post('/admin/blacklist', data),
    delBlacklist: (id) => api.delete(`/admin/blacklist/${id}`),

    // 测试台
    testTextMatch: (text) => api.post('/admin/test/text_match', null, { params: { text } })
};