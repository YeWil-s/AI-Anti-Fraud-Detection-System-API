/**
 * 管理后台调用的后端 API 根地址（须包含 /api 前缀，与 main.py 中路由一致）。
 *
 * 请使用部署方自己的后端公网/内网地址，不要使用内网穿透示例域名。
 *
 * 优先级：
 * 1. window.__ADMIN_API_BASE__（在 index.html 中设置，用于前后端不同域时）
 * 2. 同域部署：管理页与 API 在同一 origin 时，自动使用「当前站点 + /api」
 * 3. 本地：用 file:// 或 localhost 非 8000 端口打开静态页时，默认 http://localhost:8000/api
 */
function resolveDefaultApiBase() {
    if (typeof window === 'undefined') return 'http://localhost:8000/api';
    const loc = window.location;
    if (loc.protocol === 'file:') return 'http://localhost:8000/api';
    const isLocal = loc.hostname === 'localhost' || loc.hostname === '127.0.0.1';
    const p = loc.port;
    if (isLocal && p && p !== '8000') return 'http://localhost:8000/api';
    return new URL('/api', loc.origin).href;
}

export const API_BASE =
    (typeof window !== 'undefined' && window.__ADMIN_API_BASE__) || resolveDefaultApiBase();
