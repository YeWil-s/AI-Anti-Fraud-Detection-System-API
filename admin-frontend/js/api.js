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
    // =======================
    // 仪表盘与统计
    // =======================
    getStats: () => api.get('/admin/stats'),
    getTrendStats: (days = 7) => api.get('/admin/stats/trends', { params: { days } }),
    getFraudTypeStats: () => api.get('/admin/stats/fraud-types'),
    getHourlyStats: () => api.get('/admin/stats/hourly'),
    
    // =======================
    // 规则管理
    // =======================
    getRules: () => api.get('/admin/rules'),
    addRule: (data) => api.post('/admin/rules', data),
    delRule: (id) => api.delete(`/admin/rules/${id}`),

    // =======================
    // 黑名单管理
    // =======================
    getBlacklist: () => api.get('/admin/blacklist'),
    addBlacklist: (data) => api.post('/admin/blacklist', data),
    delBlacklist: (id) => api.delete(`/admin/blacklist/${id}`),

    // =======================
    // 测试台
    // =======================
    testTextMatch: (text) => api.post('/admin/test/text_match', null, { params: { text } }),

    // =======================
    // 诈骗案例学习管理
    // =======================
    getFraudCases: () => api.get('/admin/fraud-cases'),
    learnCase: (callId) => api.post(`/admin/fraud-cases/${callId}/learn`),
    learnCaseWithEdit: (callId, data) => api.post(`/admin/fraud-cases/${callId}/learn-with-edit`, data),
    
    // =======================
    // 案例上传与管理 (新增)
    // =======================
    uploadCase: (data) => api.post('/admin/cases/upload', data),
    getPendingCases: () => api.get('/admin/cases/pending'),
    getLearnedCases: () => api.get('/admin/cases/learned'),
    getPendingCaseDetail: (filename) => api.get(`/admin/cases/pending/${filename}`),
    deletePendingCase: (filename) => api.delete(`/admin/cases/pending/${filename}`),
    
    // =======================
    // 系统监控与日志
    // =======================
    getSystemLogs: (params = {}) => api.get('/admin/system/logs', { params }),
    getSystemHealth: () => api.get('/admin/system/health'),
    getRecentDetections: (limit = 20) => api.get('/admin/detection/recent', { params: { limit } }),
    
    // =======================
    // 用户管理
    // =======================
    getUsers: () => api.get('/admin/users'),
    getUser: (id) => api.get(`/admin/users/${id}`),
    updateUser: (id, data) => api.put(`/admin/users/${id}`, data),
    updateUserStatus: (id, isActive) => api.patch(`/admin/users/${id}/status`, { is_active: isActive }),
    deleteUser: (id) => api.delete(`/admin/users/${id}`),
    getUserCallStats: (id) => api.get(`/admin/users/${id}/call-stats`),
    getFamilyGroups: () => api.get('/admin/family-groups')
};