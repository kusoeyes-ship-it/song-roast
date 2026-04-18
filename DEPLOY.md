# 🚀 辣评系统部署指南

## 一、获取腾讯云密钥（混元 LLM 用）

1. 打开 https://console.cloud.tencent.com/cam/capi
2. 点击「新建密钥」（如果已有就直接复制）
3. 记下 **SecretId** 和 **SecretKey**

> ⚠️ hunyuan-lite 模型完全免费，不产生费用

---

## 二、一键部署到 Render（3 分钟搞定）

### 方法 1：Blueprint 自动部署（推荐）

1. 打开 https://dashboard.render.com
2. 注册/登录（支持 GitHub 登录）
3. 点击右上角 **New** → **Blueprint**
4. 选择仓库 `kusoeyes-ship-it/song-roast`
5. Render 会自动读取 `render.yaml` 配置
6. 在环境变量页面填入：
   - `HUNYUAN_SECRET_ID` = 你的 SecretId
   - `HUNYUAN_SECRET_KEY` = 你的 SecretKey
7. 点击 **Apply** → 等待部署完成（约 2-3 分钟）

### 方法 2：手动创建 Web Service

1. 打开 https://dashboard.render.com
2. 点击 **New** → **Web Service**
3. 连接 GitHub → 选择 `song-roast` 仓库
4. 配置：
   - **Name**: `song-roast-backend`
   - **Root Directory**: `backend`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python3 app.py`
   - **Plan**: Free
5. 在 **Environment Variables** 中添加：
   | Key | Value |
   |-----|-------|
   | `PORT` | `10000` |
   | `LLM_MODE` | `hunyuan` |
   | `HUNYUAN_SECRET_ID` | 你的 SecretId |
   | `HUNYUAN_SECRET_KEY` | 你的 SecretKey |
   | `HUNYUAN_MODEL` | `hunyuan-lite` |
6. 点击 **Create Web Service**

---

## 三、部署完成后

部署成功后，Render 会给你一个地址，格式类似：
```
https://song-roast-backend.onrender.com
```

### 验证后端
访问 `https://song-roast-backend.onrender.com/api/health`
应该看到：
```json
{"status":"ok","version":"2.1","llm_mode":"hunyuan","hunyuan_configured":true,"model":"hunyuan-lite"}
```

### 前端已自动对接
GitHub Pages 上的前端页面（https://kusoeyes-ship-it.github.io/song-roast/）
已经配置为连接 `https://song-roast-backend.onrender.com`

> 如果 Render 服务名不是 `song-roast-backend`，需要告诉我实际域名，我来更新前端配置。

---

## 四、注意事项

- **Render 免费版**会在 15 分钟无请求后休眠，首次访问需要约 30 秒唤醒
- **混元 hunyuan-lite**完全免费，无调用限制
- 如需更快响应，可升级 Render 付费版（$7/月）或换用其他平台
