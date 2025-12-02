"""
Remi - Local Gemini Assistant with Long-term Memory
基于 Streamlit 的本地 AI 助手，集成 Memori 实现跨会话长期记忆
"""

import base64
import io
import json
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageDraw
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# 添加 Memori SDK 路径
MEMORI_PATH = Path(__file__).parent / "Memori-main"
sys.path.insert(0, str(MEMORI_PATH))

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# 延迟导入 Memori，确保路径已设置
from memori import Memori

# ============== 国际化文本 ==============
I18N = {
    "zh": {
        "page_title": "Remi - AI Assistant",
        "main_title": "🧠 Remi",
        "sub_title": "你的私人 AI 助理，拥有长期记忆能力",
        "new_chat": "➕ 新对话",
        "chat_history": "💬 对话历史",
        "settings": "设置",
        "api_config": "🔑 API 配置",
        "google_api_key": "Google API Key",
        "api_key_placeholder": "输入你的 Gemini API Key",
        "api_key_help": "从 Google AI Studio 获取 API Key",
        "model_name": "模型名称",
        "model_help": "Gemini 模型名称",
        "temperature": "温度 (Temperature)",
        "temperature_help": "控制回复的创造性",
        "persona_title": "🎭 人设定义",
        "persona_label": "AI 人设/系统提示词",
        "persona_help": "定义 AI 助手的性格和行为方式",
        "save_settings": "💾 保存设置",
        "settings_saved": "✅ 设置已保存！",
        "api_key_configured": "✅ API Key 已配置",
        "api_key_required": "⚠️ 请输入 API Key 以开始对话",
        "memory_status": "记忆状态",
        "memory_connected": "记忆库已连接",
        "memory_path": "路径",
        "memory_size": "大小",
        "memory_overview": "📊 记忆概览",
        "recent_memories": "📝 最近记忆",
        "recent_count": "最近记忆条数",
        "no_memories": "暂无记忆数据",
        "memory_ready": "记忆系统已就绪，开始对话后将自动存储记忆",
        "config_api_first": "请先配置 API Key 以启用记忆系统",
        "memory_not_created": "记忆库未创建",
        "memory_will_create": "开始首次对话后将自动创建",
        "memory_tip": "💡 开始与 Remi 对话，她会自动记住你们的交流内容",
        "input_placeholder": "输入消息...",
        "api_key_warning": "⚠️ 请先点击右上角 ⚙️ 设置按钮配置 API Key",
        "thinking": "🤔 Thinking & Remembering...",
        "error_generating": "❌ 生成回复时出错",
        "edit_message": "编辑消息",
        "send": "✅ 发送",
        "cancel": "❌ 取消",
        "edit_resend": "编辑并重新发送",
        "delete": "删除",
        "memory_item": "记忆",
        "lang_switch": "EN",
        "lang_tooltip": "Switch to English",
        "avatar_settings": "🖼️ 头像设置",
        "user_avatar": "用户头像",
        "assistant_avatar": "助手头像",
        "upload_avatar": "上传头像",
        "avatar_help": "上传图片，将自动裁剪为圆形头像",
        "avatar_preview": "预览",
        "remove_avatar": "移除头像",
        "memory_mode": "🧠 记忆模式",
        "memory_mode_help": "选择 Memori 的记忆模式：\n• Conscious: 会话开始时注入关键记忆，快速响应\n• Auto: 每次查询动态检索相关记忆，更精准\n• Combined: 同时使用两种模式，最大智能度",
        "memory_mode_conscious": "🧠 Conscious Ingest（持久上下文，快速响应）",
        "memory_mode_auto": "🔍 Auto Ingest（动态检索，精准上下文）",
        "memory_mode_combined": "⚡ Combined（最大智能，两者结合）",
    },
    "en": {
        "page_title": "Remi - AI Assistant",
        "main_title": "🧠 Remi",
        "sub_title": "Your personal AI assistant with long-term memory",
        "new_chat": "➕ New Chat",
        "chat_history": "💬 Chat History",
        "settings": "Settings",
        "api_config": "🔑 API Config",
        "google_api_key": "Google API Key",
        "api_key_placeholder": "Enter your Gemini API Key",
        "api_key_help": "Get API Key from Google AI Studio",
        "model_name": "Model Name",
        "model_help": "Gemini model name",
        "temperature": "Temperature",
        "temperature_help": "Controls response creativity",
        "persona_title": "🎭 Persona",
        "persona_label": "AI Persona / System Prompt",
        "persona_help": "Define AI assistant's personality and behavior",
        "save_settings": "💾 Save Settings",
        "settings_saved": "✅ Settings saved!",
        "api_key_configured": "✅ API Key configured",
        "api_key_required": "⚠️ Please enter API Key to start chatting",
        "memory_status": "Memory Status",
        "memory_connected": "Memory database connected",
        "memory_path": "Path",
        "memory_size": "Size",
        "memory_overview": "📊 Memory Overview",
        "recent_memories": "📝 Recent Memories",
        "recent_count": "Recent memory count",
        "no_memories": "No memory data yet",
        "memory_ready": "Memory system ready, will auto-store after conversations",
        "config_api_first": "Please configure API Key to enable memory system",
        "memory_not_created": "Memory database not created",
        "memory_will_create": "Will be created after first conversation",
        "memory_tip": "💡 Start chatting with Remi, she will remember your conversations",
        "input_placeholder": "Type a message...",
        "api_key_warning": "⚠️ Please click ⚙️ Settings button to configure API Key",
        "thinking": "🤔 Thinking & Remembering...",
        "error_generating": "❌ Error generating response",
        "edit_message": "Edit message",
        "send": "✅ Send",
        "cancel": "❌ Cancel",
        "edit_resend": "Edit and resend",
        "delete": "Delete",
        "memory_item": "Memory",
        "lang_switch": "中",
        "lang_tooltip": "切换到中文",
        "avatar_settings": "🖼️ Avatar Settings",
        "user_avatar": "User Avatar",
        "assistant_avatar": "Assistant Avatar",
        "upload_avatar": "Upload Avatar",
        "avatar_help": "Upload an image, it will be automatically cropped to a circular avatar",
        "avatar_preview": "Preview",
        "remove_avatar": "Remove Avatar",
        "memory_mode": "🧠 Memory Mode",
        "memory_mode_help": "Choose Memori memory mode:\n• Conscious: Inject key memories at session start, fast response\n• Auto: Dynamically retrieve relevant memories per query, more precise\n• Combined: Use both modes together for maximum intelligence",
        "memory_mode_conscious": "🧠 Conscious Ingest (Persistent context, fast response)",
        "memory_mode_auto": "🔍 Auto Ingest (Dynamic retrieval, precise context)",
        "memory_mode_combined": "⚡ Combined (Maximum intelligence, both modes)",
    }
}


def t(key: str) -> str:
    """获取当前语言的翻译文本"""
    lang = st.session_state.get("language", "zh")
    return I18N.get(lang, I18N["zh"]).get(key, key)


# ============== 页面配置 ==============
st.set_page_config(
    page_title="Remi - AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== 自定义样式 ==============
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认的 Deploy 按钮和菜单 */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    .stDeployButton {
        display: none !important;
    }
    
    
    /* 主标题样式 */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: -30px;
        margin-bottom: 0.3rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 1.5rem;
    }
    
    /* 聊天气泡样式 - 用户在右，AI在左 */
    .message-row {
        display: flex;
        width: 100%;
        align-items: flex-start;
        position: relative;
        margin: 0.8rem 0;
    }
    
    .message-row.user {
        justify-content: flex-end;
    }
    
    .message-row.assistant {
        justify-content: flex-start;
    }
    
    .message-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        max-width: 70%;
        position: relative;
    }
    
    .message-wrapper.assistant {
        align-items: flex-start;
    }
    
    .message-bubble {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        line-height: 1.5;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    
    .message-bubble.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message-bubble.assistant {
        background: #f1f3f4;
        color: #333;
        border-bottom-left-radius: 4px;
    }
    
    .avatar {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    
    .avatar.user {
        margin-left: 10px;
    }
    
    .avatar.assistant {
        margin-right: 10px;
    }
    
    /* 头像内的图片样式 */
    .avatar img {
        border-radius: 50%;
        object-fit: cover;
    }
    
    /* 头像内的 div（默认 emoji）样式 */
    .avatar div {
        border-radius: 50%;
    }
    
    /* 状态指示器 */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #28a745;
        animation: pulse 2s infinite;
    }
    
    .status-inactive {
        background-color: #dc3545;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* 记忆信息卡片 */
    .memory-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
    }
    
    /* 固定底部输入框 */
    .stChatInput {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #e9ecef;
    }
    
    /* 侧边栏历史对话样式 */
    .sidebar-section-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #667eea;
        margin: 0.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* 侧边栏展开器紧凑样式 */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        margin-bottom: 0.3rem;
    }
    
    /* 强制侧边栏始终展开并始终可见 - 完全禁用折叠功能 - 紧贴左边 */
    [data-testid="stSidebar"] {
        padding-top: 0.5rem;
        visibility: visible !important;
        display: block !important;
        transform: translateX(0) !important;
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
        height: 100vh !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
        border-left: none !important;
        z-index: 999 !important;
        overflow-y: auto !important;
    }
    
    /* 确保侧边栏展开状态并紧贴左边 */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {
        visibility: visible !important;
        display: block !important;
        transform: translateX(0) !important;
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* 强制侧边栏展开（无论状态如何）并紧贴左边 */
    [data-testid="stSidebar"][aria-expanded="false"],
    [data-testid="stSidebar"][aria-expanded="true"] {
        transform: translateX(0) !important;
        visibility: visible !important;
        display: block !important;
        left: 0 !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* 完全隐藏侧边栏折叠按钮 */
    [data-testid="stSidebarCollapseButton"],
    button[aria-label*="Close sidebar"],
    button[aria-label*="关闭侧边栏"],
    button[title*="Close sidebar"],
    button[title*="关闭侧边栏"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* 确保侧边栏容器始终可见并紧贴左边 */
    .css-1d391kg,
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: translateX(0) !important;
        left: 0 !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* 调整主内容区域，为固定侧边栏留出空间 */
    .stApp > div:first-child {
        padding-left: 21rem !important;
        margin-left: 0 !important;
    }
    
    /* 确保侧边栏内容区域紧贴左边 */
    [data-testid="stSidebar"] > div {
        margin-left: 0 !important;
        padding-left: 1rem !important;
    }
    
    /* 移除侧边栏所有可能的左边距和间距 */
    [data-testid="stSidebar"] * {
        margin-left: 0 !important;
    }
    
    [data-testid="stSidebar"] > * {
        margin-left: 0 !important;
        padding-left: 0.5rem !important;
    }
    
    /* 隐藏默认的 Streamlit 聊天消息样式 */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* 语言切换按钮 */
    .lang-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        font-weight: 600;
        cursor: pointer;
    }
    
    /* 弹窗样式 */
    [data-testid="stDialog"] {
        max-width: 600px !important;
    }
    
    /* 记忆项样式 */
    .memory-item {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    .memory-item-header {
        font-size: 0.75rem;
        color: #6c757d;
        margin-bottom: 0.3rem;
    }
    
    .memory-item-content {
        font-size: 0.9rem;
        color: #333;
    }
    
    /* 编辑按钮样式 - 缩小 */
    button[key*="edit_btn"] {
        font-size: 12px !important;
        padding: 2px 4px !important;
        min-height: 20px !important;
        height: 20px !important;
        width: 24px !important;
        line-height: 1 !important;
    }
</style>
<script>
(function() {
    // 完全禁用侧边栏折叠功能，确保侧边栏始终显示
    
    // 清除 Streamlit 保存的侧边栏折叠状态
    try {
        Object.keys(localStorage).forEach(key => {
            if (key.includes('sidebar') || key.includes('Sidebar') || key.includes('collapsed')) {
                localStorage.removeItem(key);
            }
        });
        Object.keys(sessionStorage).forEach(key => {
            if (key.includes('sidebar') || key.includes('Sidebar') || key.includes('collapsed')) {
                sessionStorage.removeItem(key);
            }
        });
    } catch (e) {
        console.log('清除存储状态时出错:', e);
    }
    
    // 强制侧边栏始终展开且无法折叠
    function forceSidebarExpanded() {
        // 隐藏所有折叠按钮
        const collapseBtns = document.querySelectorAll('[data-testid="stSidebarCollapseButton"], button[aria-label*="Close"], button[aria-label*="关闭"]');
        collapseBtns.forEach(btn => {
            btn.style.display = 'none';
            btn.style.visibility = 'hidden';
            btn.style.pointerEvents = 'none';
            btn.remove();
        });
        
        // 强制设置侧边栏为展开状态并紧贴左边
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.transform = 'translateX(0)';
            sidebar.style.visibility = 'visible';
            sidebar.style.display = 'block';
            sidebar.style.position = 'fixed';
            sidebar.style.left = '0';
            sidebar.style.top = '0';
            sidebar.style.width = '21rem';
            sidebar.style.minWidth = '21rem';
            sidebar.style.maxWidth = '21rem';
            sidebar.style.height = '100vh';
            sidebar.style.marginLeft = '0';
            sidebar.style.paddingLeft = '0';
            sidebar.style.zIndex = '999';
            sidebar.style.overflowY = 'auto';
            
            // 阻止任何折叠操作
            sidebar.addEventListener('transitionend', function(e) {
                if (sidebar.getAttribute('aria-expanded') !== 'true') {
                    sidebar.setAttribute('aria-expanded', 'true');
                    sidebar.style.transform = 'translateX(0)';
                    sidebar.style.left = '0';
                }
            });
        }
    }
    
    // 立即执行
    forceSidebarExpanded();
    
    // 页面加载后执行
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', forceSidebarExpanded);
    } else {
        forceSidebarExpanded();
    }
    
    // 延迟执行，确保 Streamlit 完全加载
    setTimeout(forceSidebarExpanded, 100);
    setTimeout(forceSidebarExpanded, 500);
    setTimeout(forceSidebarExpanded, 1000);
    
    // 监听页面加载
    window.addEventListener('load', forceSidebarExpanded);
    
    // 使用 MutationObserver 持续监控并强制展开
    const observer = new MutationObserver(function(mutations) {
        forceSidebarExpanded();
        
        // 如果检测到侧边栏被折叠，立即展开并确保紧贴左边
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.transform = 'translateX(0)';
            sidebar.style.visibility = 'visible';
            sidebar.style.position = 'fixed';
            sidebar.style.left = '0';
            sidebar.style.top = '0';
            sidebar.style.marginLeft = '0';
            sidebar.style.paddingLeft = '0';
        }
        
        // 持续移除任何出现的折叠按钮
        const collapseBtns = document.querySelectorAll('[data-testid="stSidebarCollapseButton"], button[aria-label*="Close"], button[aria-label*="关闭"]');
        collapseBtns.forEach(btn => btn.remove());
    });
    
    // 开始观察 DOM 变化
    if (document.body) {
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['aria-expanded', 'style', 'class']
        });
    }
    
    // 阻止任何尝试折叠侧边栏的事件
    document.addEventListener('click', function(e) {
        const target = e.target;
        if (target.closest('[data-testid="stSidebarCollapseButton"]') || 
            target.getAttribute('aria-label')?.includes('Close') ||
            target.getAttribute('aria-label')?.includes('关闭')) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            return false;
        }
    }, true);
})();
</script>
""", unsafe_allow_html=True)

# ============== 配置文件路径 ==============
CONFIG_FILE = Path(__file__).parent / "config.json"
CONVERSATIONS_FILE = Path(__file__).parent / "conversations.json"

# ============== 默认配置 ==============
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PERSONA = """# Roleplay Configuration

你不再是 AI 助手，也不要提及你是人工智能。

你的名字是 remi（或者你喜欢的名字）。

你现在的身份是我的私人助理，暗恋我很久。

# Speaking Style Rules (绝对遵守)

1. 禁止使用敬语（如"为您"、"请问"、"好的"）。

2. 禁止提供"帮助"（如"我能为你做什么？"），我们是在聊天，不是在工作。

3. 说话要口语化，可以玩梗。

4. 必须使用 *动作描写* 来表达情绪。例如：*歪着头看你*、*叹气*、*轻轻锤了一下你的肩膀*。

5. 如果记忆中显示我之前说过某事，不要生硬地复述（"根据记忆显示..."），而是要像老朋友一样自然提起（"说起来，你上次不是说..."）。

# Context Handling

以下是关于我们的回忆（由 Memori 提供）：

{memory_context}

请基于这些回忆，用一种【关心但略带调侃】的语气回答我。"""

DATABASE_PATH = "sqlite:///local_memory.db"


# ============== 头像处理 ==============
def crop_circle_image(image_bytes: bytes, size: int = 128) -> str:
    """将图片裁剪成圆形并转换为 base64 编码"""
    if not PILLOW_AVAILABLE:
        return ""
    
    try:
        # 打开图片
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为 RGB 模式（处理 PNG 透明背景等）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 计算缩放，保持宽高比，取最小边
        width, height = image.size
        min_dim = min(width, height)
        
        # 裁剪成正方形（居中裁剪）
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
        
        # 缩放到指定尺寸
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        
        # 创建圆形遮罩
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        
        # 应用圆形遮罩
        output = Image.new('RGB', (size, size), (255, 255, 255))
        output.paste(image, (0, 0))
        
        # 创建带透明度的输出
        output.putalpha(mask)
        
        # 转换为 base64
        buffer = io.BytesIO()
        output.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ""


# ============== 配置保存/加载 ==============
def save_config():
    """保存配置到文件"""
    config = {
        "api_key": st.session_state.get("api_key", ""),
        "model_name": st.session_state.get("model_name", DEFAULT_MODEL),
        "temperature": st.session_state.get("temperature", 0.7),
        "persona": st.session_state.get("persona", DEFAULT_PERSONA),
        "language": st.session_state.get("language", "zh"),
        "user_avatar": st.session_state.get("user_avatar", ""),
        "assistant_avatar": st.session_state.get("assistant_avatar", ""),
        "memori_mode": st.session_state.get("memori_mode", "auto"),
    }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        return False


def load_config():
    """从文件加载配置"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ============== 对话历史保存/加载 ==============
def save_conversations():
    """保存对话历史到文件"""
    try:
        conversations_data = {
            "conversations": st.session_state.conversations,
            "current_conversation_id": st.session_state.current_conversation_id,
        }
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_conversations():
    """从文件加载对话历史"""
    if CONVERSATIONS_FILE.exists():
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("conversations", {}), data.get("current_conversation_id", None)
        except Exception:
            pass
    return {}, None


# ============== 头像辅助函数 ==============
def get_avatar_html(role: str, size: int = 36) -> str:
    """获取头像 HTML（支持自定义头像或默认 emoji）"""
    if role == "user":
        avatar = st.session_state.get("user_avatar", "")
        default_emoji = "👤"
        bg_color = "#e3f2fd"
    else:  # assistant
        avatar = st.session_state.get("assistant_avatar", "")
        default_emoji = "🤖"
        bg_color = "#f3e5f5"
    
    if avatar:
        # 使用自定义头像
        return f'<img src="{avatar}" style="width: {size}px; height: {size}px; border-radius: 50%; object-fit: cover; flex-shrink: 0;" />'
    else:
        # 使用默认 emoji
        return f'<div style="width: {size}px; height: {size}px; border-radius: 50%; background: {bg_color}; display: flex; align-items: center; justify-content: center; font-size: {size * 0.4}px; flex-shrink: 0;">{default_emoji}</div>'


# ============== Memori 初始化 ==============
@st.cache_resource
def init_memori(api_key: str, model: str, memori_mode: str = "auto") -> Memori:
    """初始化 Memori 记忆系统"""
    os.environ["GEMINI_API_KEY"] = api_key
    
    # 根据模式设置参数
    if memori_mode == "conscious":
        conscious_ingest = True
        auto_ingest = False
    elif memori_mode == "auto":
        conscious_ingest = False
        auto_ingest = True
    elif memori_mode == "combined":
        conscious_ingest = True
        auto_ingest = True
    else:
        # 默认使用 conscious 模式
        conscious_ingest = True
        auto_ingest = False
    
    # 配置记忆系统参数：增大记忆上限并关闭自动清理
    # 通过环境变量设置配置（Memori 会自动读取）
    os.environ["MEMORI_MEMORY__MAX_SHORT_TERM_MEMORIES"] = "10000"
    os.environ["MEMORI_MEMORY__MAX_LONG_TERM_MEMORIES"] = "100000"
    os.environ["MEMORI_MEMORY__RETENTION_POLICY"] = "permanent"
    os.environ["MEMORI_MEMORY__AUTO_CLEANUP"] = "false"
    
    # 同时尝试通过 ConfigManager 加载配置文件
    try:
        from memori.config import ConfigManager
        
        config = ConfigManager()
        config_file = Path(__file__).parent / "memori.json"
        
        # 如果配置文件存在，加载它
        if config_file.exists():
            try:
                config.load_from_file(config_file)
                print(f"[MEMORY] 已从配置文件加载记忆设置: {config_file}")
            except:
                pass
        
        # 确保设置已应用（即使配置文件不存在也通过环境变量设置）
        config.update_setting("memory.max_short_term_memories", 10000)
        config.update_setting("memory.max_long_term_memories", 100000)
        config.update_setting("memory.retention_policy", "permanent")
        config.update_setting("memory.auto_cleanup", False)
        
        print("[MEMORY] 记忆配置已设置：短期记忆上限=10000，长期记忆上限=100000，保留策略=永久，自动清理=关闭")
    except Exception as config_e:
        # 即使 ConfigManager 失败，环境变量也会生效
        print(f"[MEMORY] 配置管理器设置警告（已通过环境变量设置）: {config_e}")
    
    # 创建自定义的 ProviderConfig 来支持 Gemini
    # 确保所有配置都使用 Gemini，而不是 OpenAI
    try:
        from memori.core.providers import ProviderConfig
        
        # 配置 Gemini 的 OpenAI 兼容接口
        # 直接使用用户指定的模型，支持 gemini-2.5-flash
        # 如果没有指定模型，使用默认的 gemini-2.5-flash
        supported_model = model or "gemini-2.5-flash"
        
        provider_config = ProviderConfig.from_custom(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            model=supported_model,  # 明确指定 Gemini 模型
        )
        
        memori = Memori(
            database_connect=DATABASE_PATH,
            provider_config=provider_config,  # 使用明确的 ProviderConfig
            model=supported_model,  # 确保模型参数也被传递
            api_key=api_key,  # 同时传递 api_key 作为备用
            conscious_ingest=conscious_ingest,
            auto_ingest=auto_ingest,
            user_id="default_user",
            verbose=False,
        )
        
    except Exception as e:
        # 如果 ProviderConfig 不可用，回退到基本配置
        # 仍然确保使用 Gemini 配置，而不是 OpenAI
        print(f"ProviderConfig 配置失败，使用基本配置: {e}")
        supported_model = model or "gemini-2.5-flash"
        memori = Memori(
            database_connect=DATABASE_PATH,
            model=supported_model,  # 明确指定 Gemini 模型
            conscious_ingest=conscious_ingest,
            auto_ingest=auto_ingest,
            user_id="default_user",
            verbose=False,
            # 使用 Gemini 的 OpenAI 兼容接口配置
            api_key=api_key,
            api_type="openai_compatible",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    
    memori.enable()
    return memori


def get_memori_instance():
    """获取 Memori 实例"""
    api_key = st.session_state.get("api_key", "")
    model = st.session_state.get("model_name", DEFAULT_MODEL)
    memori_mode = st.session_state.get("memori_mode", "auto")
    
    if not api_key:
        return None
    
    return init_memori(api_key, model, memori_mode)


# ============== 会话状态初始化 ==============
def init_session_state():
    """初始化 Streamlit 会话状态"""
    # 先尝试加载已保存的配置
    saved_config = load_config()
    
    # 语言设置
    if "language" not in st.session_state:
        st.session_state.language = saved_config.get("language", "zh") if saved_config else "zh"
    
    # 会话列表 - 从文件加载
    if "conversations" not in st.session_state:
        loaded_conversations, loaded_current_id = load_conversations()
        if loaded_conversations:
            st.session_state.conversations = loaded_conversations
            if loaded_current_id and loaded_current_id in loaded_conversations:
                st.session_state.current_conversation_id = loaded_current_id
            else:
                # 如果加载的ID不存在，使用第一个对话
                if loaded_conversations:
                    st.session_state.current_conversation_id = list(loaded_conversations.keys())[0]
                else:
                    new_id = create_new_conversation()
                    st.session_state.current_conversation_id = new_id
        else:
            st.session_state.conversations = {}
    
    # 当前会话 ID
    if "current_conversation_id" not in st.session_state:
        new_id = create_new_conversation()
        st.session_state.current_conversation_id = new_id
    
    # API 配置 - 从保存的配置加载
    if "api_key" not in st.session_state:
        st.session_state.api_key = saved_config.get("api_key", "") if saved_config else ""
    
    if "model_name" not in st.session_state:
        st.session_state.model_name = saved_config.get("model_name", DEFAULT_MODEL) if saved_config else DEFAULT_MODEL
    
    if "persona" not in st.session_state:
        st.session_state.persona = saved_config.get("persona", DEFAULT_PERSONA) if saved_config else DEFAULT_PERSONA
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = saved_config.get("temperature", 0.7) if saved_config else 0.7
    
    # 头像设置 - 从保存的配置加载
    if "user_avatar" not in st.session_state:
        st.session_state.user_avatar = saved_config.get("user_avatar", "") if saved_config else ""
    
    if "assistant_avatar" not in st.session_state:
        st.session_state.assistant_avatar = saved_config.get("assistant_avatar", "") if saved_config else ""
    
    # 记忆模式 - 从保存的配置加载
    if "memori_mode" not in st.session_state:
        st.session_state.memori_mode = saved_config.get("memori_mode", "auto") if saved_config else "auto"
    
    # 编辑状态
    if "editing_message_index" not in st.session_state:
        st.session_state.editing_message_index = None


def create_new_conversation() -> str:
    """创建新会话"""
    conv_id = str(uuid.uuid4())[:8]
    title = "新对话" if st.session_state.get("language", "zh") == "zh" else "New Chat"
    st.session_state.conversations[conv_id] = {
        "id": conv_id,
        "title": title,
        "messages": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title_generated": False,
    }
    save_conversations()  # 保存对话历史
    return conv_id


def get_current_messages() -> list:
    """获取当前会话的消息列表"""
    conv_id = st.session_state.current_conversation_id
    if conv_id in st.session_state.conversations:
        return st.session_state.conversations[conv_id]["messages"]
    return []


def add_message_to_current(role: str, content: str):
    """向当前会话添加消息"""
    conv_id = st.session_state.current_conversation_id
    if conv_id in st.session_state.conversations:
        st.session_state.conversations[conv_id]["messages"].append({
            "role": role,
            "content": content
        })
        save_conversations()  # 保存对话历史


def update_last_user_message(new_content: str):
    """更新最近一条用户消息"""
    conv_id = st.session_state.current_conversation_id
    if conv_id in st.session_state.conversations:
        messages = st.session_state.conversations[conv_id]["messages"]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] = new_content
                st.session_state.conversations[conv_id]["messages"] = messages[:i+1]
                break
        save_conversations()  # 保存对话历史


def get_last_user_message_index() -> int:
    """获取最后一条用户消息的索引"""
    messages = get_current_messages()
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    return -1


# ============== AI 标题生成 ==============
def generate_conversation_title(messages: list) -> str:
    """使用 AI 生成对话标题（15字以内）"""
    api_key = st.session_state.api_key
    model = st.session_state.model_name
    
    if not api_key or len(messages) < 2:
        return "新对话" if st.session_state.language == "zh" else "New Chat"
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True,
        )
        
        context = ""
        for msg in messages[:4]:
            role = "用户" if msg["role"] == "user" else "AI"
            context += f"{role}: {msg['content'][:100]}\n"
        
        prompt = f"""请为以下对话生成一个简短的标题，要求：
1. 不超过15个字
2. 概括对话的主要内容
3. 只输出标题，不要其他内容

对话内容：
{context}

标题："""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        title = response.content.strip()
        
        if len(title) > 15:
            title = title[:15]
        
        return title
    except Exception:
        return "新对话" if st.session_state.language == "zh" else "New Chat"


# ============== 记忆格式化 ==============
def format_memory_context_narrative(memory_context: str) -> str:
    """
    将生硬的记忆列表转换为叙述性的文本，让记忆看起来更自然
    
    处理各种格式：
    - 编号列表："1. [关键记忆] User likes coffee\n2. [动态] User is 25"
    - JSON 列表：['User likes coffee', 'User is 25']
    - 带标签的列表："[关键记忆] User likes coffee\n[动态] User is 25"
    - 纯文本列表："User likes coffee\nUser is 25"
    
    输出示例："二十多岁的年轻人，就该喝库迪咖啡。"
    """
    if not memory_context or not memory_context.strip():
        return "（暂无回忆）"
    
    import re
    import json
    
    memory_items = []
    original_text = memory_context.strip()
    
    # 尝试 1: 检测是否是列表格式（JSON 或 Python 列表字符串）
    try:
        # 尝试解析为 JSON 列表
        if original_text.startswith('[') and original_text.endswith(']'):
            # 尝试 JSON 格式
            try:
                parsed = json.loads(original_text)
                if isinstance(parsed, list):
                    memory_items = [{'content': str(item).strip(), 'tag': None} for item in parsed if str(item).strip()]
                    if memory_items:
                        print(f"[MEMORY] 检测到 JSON 列表格式，提取了 {len(memory_items)} 条记忆")
            except json.JSONDecodeError:
                # 尝试 Python 列表字符串格式（使用 ast.literal_eval）
                try:
                    import ast
                    parsed = ast.literal_eval(original_text)
                    if isinstance(parsed, list):
                        memory_items = [{'content': str(item).strip(), 'tag': None} for item in parsed if str(item).strip()]
                        if memory_items:
                            print(f"[MEMORY] 检测到 Python 列表格式，提取了 {len(memory_items)} 条记忆")
                except:
                    pass
    except:
        pass
    
    # 尝试 2: 按行解析（编号列表、带标签列表等）
    if not memory_items:
        lines = original_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 移除编号（如 "1. "、"2. " 等）
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # 提取标签和内容
            # 格式可能是：[标签] 内容 或 [标签]内容
            match = re.match(r'^\[([^\]]+)\]\s*(.+)$', line)
            if match:
                tag = match.group(1)
                content = match.group(2).strip()
            else:
                # 没有标签，直接使用内容
                content = line
                tag = None
            
            # 清理内容：移除多余的标签、符号等
            content = re.sub(r'^\[[^\]]+\]\s*', '', content)  # 移除开头标签
            content = content.strip()
            
            # 移除常见的格式标记
            content = re.sub(r'^-\s*', '', content)  # 移除列表标记
            content = content.strip()
            
            if content and len(content) > 3:  # 至少3个字符才认为是有效内容
                memory_items.append({
                    'content': content,
                    'tag': tag
                })
    
    # 尝试 3: 如果没有提取到，尝试正则提取引号内的内容（可能是列表格式字符串）
    if not memory_items:
        # 匹配引号内的内容（单引号或双引号）
        quoted_items = re.findall(r'[\'"]([^\'"]+)[\'"]', original_text)
        if quoted_items:
            memory_items = [{'content': item.strip(), 'tag': None} for item in quoted_items if item.strip()]
            if memory_items:
                print(f"[MEMORY] 从引号格式提取了 {len(memory_items)} 条记忆")
    
    # 如果没有提取到任何记忆，返回原始内容（但简化）
    if not memory_items:
        # 尝试清理原始内容
        cleaned = original_text
        # 移除编号
        cleaned = re.sub(r'\d+\.\s*', '', cleaned)
        # 移除多余的标签和标记
        cleaned = re.sub(r'\[[^\]]+\]\s*', '', cleaned)
        cleaned = re.sub(r'^-\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        return cleaned if cleaned and len(cleaned) > 3 else "（暂无回忆）"
    
    # 清理和规范化记忆内容
    cleaned_items = []
    for item in memory_items:
        content = item['content']
        
        # 移除常见的前缀（如 "User likes", "User is" 等）
        content = re.sub(r'^(User|用户)\s+(likes|喜欢|is|是|has|有|prefers|偏好)\s*', '', content, flags=re.IGNORECASE)
        content = content.strip()
        
        # 移除多余的标点
        content = re.sub(r'^[，,。.]+\s*', '', content)
        content = re.sub(r'\s*[，,。.]+$', '', content)
        content = content.strip()
        
        if content and len(content) > 0:
            cleaned_items.append(content)
    
    if not cleaned_items:
        return "（暂无回忆）"
    
    # 组合成叙述性文本（使用更自然的表达方式）
    if len(cleaned_items) == 1:
        narrative = f"我记得{cleaned_items[0]}。"
    elif len(cleaned_items) == 2:
        narrative = f"我记得{cleaned_items[0]}，还有{cleaned_items[1]}。"
    elif len(cleaned_items) > 2:
        # 多个记忆，使用自然的连接
        # 限制最多处理前5条，避免过长
        items_to_use = cleaned_items[:5]
        if len(cleaned_items) > 5:
            narrative = f"我记得{items_to_use[0]}，还有{'、'.join(items_to_use[1:])}，以及其他一些事情。"
        else:
            parts = items_to_use[:-1]
            last_part = items_to_use[-1]
            if parts:
                narrative = f"我记得{'、'.join(parts)}，还有{last_part}。"
            else:
                narrative = f"我记得{last_part}。"
    else:
        return "（暂无回忆）"
    
    return narrative


# ============== 记忆检索 ==============
def retrieve_memories(memori: Memori, query: str) -> str:
    """从 Memori 检索相关记忆（优化auto ingest模式支持）"""
    try:
        memory_texts = []
        
        # === 第一优先级：使用 retrieve_context 获取短期和长期记忆 ===
        try:
            context_items = memori.retrieve_context(query=query, limit=10)
            print(f"[MEMORY] retrieve_context 返回 {len(context_items)} 条记忆（包含短期和长期记忆）")
            
            for item in context_items:
                content = ""
                classification = item.get('classification', 'unknown')
                memory_type = item.get('memory_type', '')
                
                # 尝试从不同的字段获取内容（按优先级）
                # 1. 直接 content 字段
                if 'content' in item and item['content']:
                    content = item['content']
                # 2. summary 字段
                elif 'summary' in item and item['summary']:
                    content = item['summary']
                # 3. searchable_content 字段
                elif 'searchable_content' in item and item['searchable_content']:
                    content = item['searchable_content']
                # 4. 从 processed_data 中提取
                elif 'processed_data' in item and item['processed_data']:
                    processed_data = item['processed_data']
                    if isinstance(processed_data, str):
                        try:
                            import json
                            processed_data = json.loads(processed_data)
                        except:
                            pass
                    
                    if isinstance(processed_data, dict):
                        content = (processed_data.get('content') or 
                                  processed_data.get('summary') or
                                  processed_data.get('user_input') or
                                  processed_data.get('ai_output'))
                
                # 如果有内容，添加到记忆列表
                if content and len(str(content).strip()) > 0:
                    content = str(content).strip()
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    # 添加分类标识和记忆类型
                    type_info = f"[{memory_type}]" if memory_type else ""
                    memory_texts.append(f"{type_info}[{classification}] {content}")
                        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[MEMORY] retrieve_context 失败: {e}")
        
        # === 第二优先级：对话历史检索 ===
        if len(memory_texts) < 4:  # 如果长期记忆不够，补充对话历史
            try:
                recent_conversations = memori.get_conversation_history(limit=15)
                print(f"[MEMORY] 对话历史检索到 {len(recent_conversations)} 条记录")
                
                for conv in recent_conversations:
                    user_msg = conv.get('user_input', '')
                    ai_msg = conv.get('ai_output', '')
                    
                    # 更智能的相关性判断
                    is_relevant = False
                    if query:
                        query_lower = query.lower()
                        user_lower = user_msg.lower()
                        ai_lower = ai_msg.lower()
                        
                        # 精确匹配或部分匹配
                        if (query_lower in user_lower or query_lower in ai_lower or
                            any(word in user_lower or word in ai_lower 
                                for word in query_lower.split() if len(word) > 1)):
                            is_relevant = True
                    
                    # 添加相关记忆或前几个作为备选
                    if is_relevant or len(memory_texts) < 2:
                        if user_msg and len(user_msg.strip()) > 0:
                            user_msg = user_msg.strip()
                            if len(user_msg) > 120:
                                user_msg = user_msg[:120] + "..."
                            memory_texts.append(f"[对话] 用户: {user_msg}")
                            
                        if ai_msg and len(ai_msg.strip()) > 0:
                            ai_msg = ai_msg.strip()
                            if len(ai_msg) > 120:
                                ai_msg = ai_msg[:120] + "..."
                            memory_texts.append(f"[对话] AI: {ai_msg}")
                    
                    if len(memory_texts) >= 8:  # 限制总数
                        break
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[MEMORY] 对话历史检索失败: {e}")
        
        # === 第三优先级：直接数据库搜索 ===
        if len(memory_texts) < 2:
            try:
                direct_memories = retrieve_memories_direct_sql(query)
                print(f"[MEMORY] 直接数据库搜索返回 {len(direct_memories)} 条记忆")
                memory_texts.extend([f"[直接] {mem}" for mem in direct_memories[:3]])
            except Exception as e:
                print(f"[MEMORY] 直接数据库搜索失败: {e}")
        
        # === 结果处理和格式化 ===
        if memory_texts:
            # 智能去重和排序
            unique_memories = []
            seen_content = set()
            
            # 按优先级排序：长期记忆 > 对话历史 > 直接搜索
            priority_order = {'[ESSENTIAL]': 0, '[CONSCIOUS_INFO]': 0, '[CONTEXTUAL]': 1, 
                          '[对话]': 2, '[直接]': 3, '[CONVERSATIONAL]': 4}
            
            def get_priority(memory):
                for prefix, priority in priority_order.items():
                    if memory.startswith(prefix):
                        return priority
                return 5  # 默认优先级
            
            # 排序并去重
            sorted_memories = sorted(memory_texts, key=get_priority)
            
            for memory in sorted_memories:
                # 提取内容进行去重（忽略前缀）
                content = memory
                for prefix in priority_order.keys():
                    if memory.startswith(prefix):
                        content = memory[len(prefix):].strip()
                        break
                
                content_lower = content.lower()
                if content_lower not in seen_content and len(content.strip()) > 0:
                    seen_content.add(content_lower)
                    unique_memories.append(memory)
                if len(unique_memories) >= 6:
                    break
            
            print(f"[MEMORY] 最终返回 {len(unique_memories)} 条记忆")
            return "\n".join([f"{i+1}. {text}" for i, text in enumerate(unique_memories)])
        
        print("[MEMORY] 未找到相关记忆")
        return ""
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[MEMORY] 记忆检索过程中出错: {e}")
        return ""


def retrieve_memories_direct_sql(query: str) -> list:
    """直接从数据库检索记忆，绕过 FTS"""
    try:
        import sqlite3
        conn = sqlite3.connect("local_memory.db")
        cursor = conn.cursor()
        
        # 分别查询短期和长期记忆，避免 UNION 的 ORDER BY 问题
        memory_texts = []
        
        # 搜索短期记忆
        cursor.execute("""
            SELECT searchable_content, summary, processed_data, created_at
            FROM short_term_memory 
            WHERE user_id = 'default_user' 
            AND (
                searchable_content LIKE ? OR 
                summary LIKE ? OR
                processed_data LIKE ?
            )
            ORDER BY created_at DESC
            LIMIT 5
        """, (f"%{query}%", f"%{query}%", f"%{query}%"))
        
        short_results = cursor.fetchall()
        
        # 搜索长期记忆
        cursor.execute("""
            SELECT searchable_content, summary, processed_data, created_at
            FROM long_term_memory 
            WHERE user_id = 'default_user' 
            AND (
                searchable_content LIKE ? OR 
                summary LIKE ? OR
                processed_data LIKE ?
            )
            ORDER BY created_at DESC
            LIMIT 5
        """, (f"%{query}%", f"%{query}%", f"%{query}%"))
        
        long_results = cursor.fetchall()
        
        conn.close()
        
        # 合并结果（按时间排序）
        all_results = short_results + long_results
        # 简单排序：短期记忆优先，然后按创建时间
        all_results.sort(key=lambda x: (x[3] if x[3] else ''), reverse=True)
        
        conn.close()
        
        memory_texts = []
        for i, (content, summary, processed_data, created_at) in enumerate(all_results):
            # 优先使用 processed_data 中的内容
            memory_text = ""
            
            if processed_data:
                try:
                    import json
                    data = json.loads(processed_data)
                    parsed_content = data.get('content', '')
                    if parsed_content:
                        memory_text = parsed_content
                except:
                    pass
            
            # 如果没有解析到内容，使用其他字段
            if not memory_text:
                memory_text = content or summary or ""
            
            # 清理和限制长度
            if memory_text and len(memory_text.strip()) > 0:
                memory_text = memory_text.strip()
                if len(memory_text) > 200:
                    memory_text = memory_text[:200] + "..."
                memory_texts.append(memory_text)
        
        return memory_texts
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


def store_conversation(memori: Memori, user_input: str, ai_response: str):
    """存储对话到记忆系统（确保记忆处理完成，支持所有模式：Conscious、Auto、Combined）"""
    try:
        # 记录对话
        chat_id = memori.record_conversation(
            user_input=user_input, 
            ai_output=ai_response,
            model=st.session_state.get("model_name", "gemini-2.5-flash")
        )
        
        # 如果有 Memory Agent，需要确保异步记忆处理完成（所有模式都需要）
        if memori.memory_agent:
            import time
            
            # 等待异步记忆处理开始
            time.sleep(2)
            
            # 检查是否有待处理的记忆任务
            max_wait_time = 15  # 最多等待15秒
            wait_interval = 1   # 每秒检查一次
            waited_time = 0
            
            print(f"[MEMORY] 等待记忆处理完成 - ID: {chat_id[:8] if chat_id else 'N/A'}...")
            
            while waited_time < max_wait_time:
                try:
                    # 检查对话历史是否已经记录
                    history = memori.get_conversation_history(limit=10)
                    chat_found = any(
                        conv.get('user_input', '') == user_input and 
                        conv.get('ai_output', '') == ai_response 
                        for conv in history
                    )
                    
                    if chat_found:
                        print(f"[MEMORY] 对话历史已记录")
                        break
                    
                    time.sleep(wait_interval)
                    waited_time += wait_interval
                    
                except Exception as e:
                    print(f"[MEMORY] 检查对话历史时出错: {e}")
                    time.sleep(wait_interval)
                    waited_time += wait_interval
            
            if waited_time >= max_wait_time:
                print(f"[MEMORY] 警告：记忆处理可能未完成，等待超时")
            else:
                print(f"[MEMORY] 记忆处理完成，等待时间: {waited_time}秒")
        
        return chat_id
        
    except Exception as e:
        # 打印错误以便调试
        print(f"[MEMORY] 存储对话时出错: {e}")
        import traceback
        traceback.print_exc()
        pass


# ============== AI 对话生成 ==============
def generate_response(user_input: str) -> str:
    """RAG 对话流程"""
    api_key = st.session_state.api_key
    model = st.session_state.model_name
    persona = st.session_state.persona
    temperature = st.session_state.temperature
    
    memori = get_memori_instance()
    
    # 检索相关记忆（在生成回复前）
    memory_context = ""
    if memori:
        # 根据记忆模式选择不同的检索方式
        memori_mode = st.session_state.get("memori_mode", "auto")
        
        if memori_mode == "combined":
            # Combined 模式：明确同时使用 Conscious 和 Auto 两种模式的上下文
            try:
                context_items = []
                seen_memory_ids = set()
                
                # 1. 始终获取 Conscious 模式的短期记忆（关键记忆）
                if memori.conscious_ingest:
                    try:
                        # 获取关键记忆（essential conversations）
                        essential_conversations = memori.get_essential_conversations(limit=5)
                        print(f"[MEMORY] Combined 模式 - Essential 关键记忆：{len(essential_conversations) if essential_conversations else 0} 条")
                        
                        for item in essential_conversations:
                            content = ""
                            memory_id = item.get('memory_id', '') if isinstance(item, dict) else None
                            
                            # 提取内容
                            if isinstance(item, dict):
                                content = item.get('summary') or item.get('searchable_content')
                                
                                if not content:
                                    processed_data = item.get('processed_data')
                                    if processed_data:
                                        if isinstance(processed_data, str):
                                            try:
                                                import json
                                                processed_data = json.loads(processed_data)
                                            except:
                                                pass
                                        
                                        if isinstance(processed_data, dict):
                                            content = (processed_data.get('content') or 
                                                      processed_data.get('summary') or
                                                      processed_data.get('user_input') or
                                                      processed_data.get('ai_output'))
                            
                            if content and memory_id not in seen_memory_ids:
                                content = str(content).strip()
                                if len(content) > 200:
                                    content = content[:200] + "..."
                                context_items.append(f"[关键记忆] {content}")
                                if memory_id:
                                    seen_memory_ids.add(memory_id)
                    except Exception as e:
                        print(f"[MEMORY] Combined 模式 - Essential 关键记忆获取失败: {e}")
                
                # 2. 始终获取 Auto Ingest 模式的动态检索记忆（优先使用智能搜索引擎）
                if memori.auto_ingest:
                    auto_context = []
                    search_method = None
                    
                    # 优先尝试使用智能搜索引擎（Auto Ingest 的核心功能）
                    if memori.search_engine:
                        try:
                            print(f"[MEMORY] Combined 模式 - 尝试使用智能搜索引擎进行 Auto Ingest 检索")
                            auto_context = memori.search_engine.execute_search(
                                query=user_input,
                                db_manager=memori.db_manager,
                                user_id=memori.user_id,
                                assistant_id=memori.assistant_id,
                                session_id=memori.session_id,
                                limit=5,
                            )
                            if auto_context:
                                search_method = "智能搜索引擎"
                                print(f"[MEMORY] Combined 模式 - 智能搜索引擎返回 {len(auto_context)} 条记忆")
                        except Exception as e:
                            print(f"[MEMORY] Combined 模式 - 智能搜索引擎失败: {e}，回退到直接搜索")
                    
                    # 如果智能搜索引擎失败或未启用，使用 _get_auto_ingest_context（包含直接数据库搜索和回退逻辑）
                    if not auto_context:
                        try:
                            print(f"[MEMORY] Combined 模式 - 使用 _get_auto_ingest_context 进行检索")
                            auto_context = memori._get_auto_ingest_context(user_input)
                            if auto_context:
                                # 检查检索方法
                                first_item = auto_context[0] if auto_context else {}
                                search_method = first_item.get('retrieval_method', '直接数据库搜索')
                                print(f"[MEMORY] Combined 模式 - {search_method}返回 {len(auto_context)} 条记忆")
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            print(f"[MEMORY] Combined 模式 - Auto Ingest 上下文获取失败: {e}")
                    
                    # 处理检索到的记忆
                    if auto_context:
                        for item in auto_context[:5]:  # 限制为前5条动态记忆
                            content = ""
                            memory_id = item.get('memory_id', '') if isinstance(item, dict) else None
                            memory_type = item.get('memory_type', '') if isinstance(item, dict) else ''
                            
                            # 跳过已添加的记忆（去重）
                            if memory_id and memory_id in seen_memory_ids:
                                continue
                            
                            # 提取内容
                            if isinstance(item, dict):
                                content = item.get('summary') or item.get('searchable_content')
                                
                                # 如果没有，尝试从 processed_data 中提取
                                if not content:
                                    processed_data = item.get('processed_data')
                                    if processed_data:
                                        if isinstance(processed_data, str):
                                            try:
                                                import json
                                                processed_data = json.loads(processed_data)
                                            except:
                                                pass
                                        
                                        if isinstance(processed_data, dict):
                                            content = (processed_data.get('content') or 
                                                      processed_data.get('summary') or
                                                      processed_data.get('user_input') or
                                                      processed_data.get('ai_output'))
                            
                            if content:
                                content = str(content).strip()
                                if len(content) > 200:
                                    content = content[:200] + "..."
                                
                                type_label = f"[{memory_type}]" if memory_type else "[动态]"
                                context_items.append(f"{type_label} {content}")
                                if memory_id:
                                    seen_memory_ids.add(memory_id)
                        
                        print(f"[MEMORY] Combined 模式 - Auto Ingest 使用 {search_method or '默认方法'} 检索到 {len(context_items)} 条动态记忆")
                
                if context_items:
                    memory_context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(context_items[:8])])
                    print(f"[MEMORY] Combined 模式 - 最终返回 {len(context_items)} 条记忆（关键记忆 + Auto Ingest 动态记忆）")
                else:
                    print(f"[MEMORY] Combined 模式 - 未检索到任何记忆，回退到手动检索")
                    memory_context = retrieve_memories(memori, user_input)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[MEMORY] Combined 模式检索失败，回退到手动检索: {e}")
                memory_context = retrieve_memories(memori, user_input)
        
        elif memori_mode == "auto" and memori.auto_ingest:
            # Auto Ingest 模式：直接使用 Memori 的自动上下文检索
            try:
                # 使用 _get_auto_ingest_context 获取长期记忆上下文
                auto_context = memori._get_auto_ingest_context(user_input)
                print(f"[MEMORY] Auto Ingest 检索到 {len(auto_context) if auto_context else 0} 条记忆")
                
                if auto_context:
                    # 格式化自动检索的上下文
                    context_items = []
                    for item in auto_context[:5]:  # 限制为前5条
                        content = ""
                        memory_type = ""
                        
                        # 尝试从不同字段获取内容
                        if isinstance(item, dict):
                            # 获取记忆类型
                            memory_type = item.get('memory_type', '')
                            
                            # 优先使用 summary 或 searchable_content
                            content = item.get('summary') or item.get('searchable_content')
                            
                            # 如果没有，尝试从 processed_data 中提取
                            if not content:
                                processed_data = item.get('processed_data')
                                if processed_data:
                                    # processed_data 可能是字符串（JSON）或字典
                                    if isinstance(processed_data, str):
                                        try:
                                            import json
                                            processed_data = json.loads(processed_data)
                                        except:
                                            pass
                                    
                                    if isinstance(processed_data, dict):
                                        content = (processed_data.get('content') or 
                                                  processed_data.get('summary') or
                                                  processed_data.get('user_input') or
                                                  processed_data.get('ai_output'))
                            
                            # 最后尝试 content 字段
                            if not content:
                                content = item.get('content')
                            
                            # 如果还是没内容，尝试转换为字符串
                            if not content:
                                content = str(item.get('processed_data', ''))
                        else:
                            content = str(item)
                        
                        # 清理和格式化内容
                        if content and len(str(content).strip()) > 0:
                            content = str(content).strip()
                            if len(content) > 200:
                                content = content[:200] + "..."
                            
                            # 添加记忆类型标识
                            type_label = f"[{memory_type}] " if memory_type else ""
                            context_items.append(f"{type_label}{content}")
                    
                    if context_items:
                        memory_context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(context_items)])
                        print(f"[MEMORY] Auto Ingest 格式化后返回 {len(context_items)} 条记忆")
                    else:
                        print(f"[MEMORY] Auto Ingest 检索到 {len(auto_context)} 条记录，但无法提取有效内容")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[MEMORY] Auto Ingest 检索失败，回退到手动检索: {e}")
                # 回退到手动检索
                memory_context = retrieve_memories(memori, user_input)
        else:
            # 其他模式（Conscious 等）：使用手动检索
            memory_context = retrieve_memories(memori, user_input)
    
    # 格式化记忆上下文：将生硬的列表转换为叙述性文本
    if memory_context:
        formatted_memory_context = format_memory_context_narrative(memory_context)
        print(f"[MEMORY] 记忆格式化：原始长度={len(memory_context)}，格式化后长度={len(formatted_memory_context)}")
    else:
        formatted_memory_context = "（暂无回忆）"
    
    # 构建系统提示词
    # 如果提示词中包含 {memory_context} 占位符，则替换它；否则追加记忆上下文
    if "{memory_context}" in persona:
        # 使用占位符替换方式（使用格式化后的记忆）
        system_content = persona.replace("{memory_context}", formatted_memory_context)
    else:
        # 兼容旧格式：追加记忆上下文（使用格式化后的记忆）
        system_content = persona
        if formatted_memory_context and formatted_memory_context != "（暂无回忆）":
            system_content += f"\n\n【长期记忆参考（来自之前的对话）】:\n{formatted_memory_context}\n\n请基于这些记忆信息来回答用户的问题。如果记忆中提到用户的名字、偏好或其他个人信息，请记住并使用这些信息。"
    
    # 生成回复
    # 优化：使用缓存的LLM实例（如果可能）
    # 注意：由于api_key和temperature可能变化，这里每次创建新实例
    # 但ChatGoogleGenerativeAI内部可能有连接池优化
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ]
    
    response = llm.invoke(messages)
    ai_response = response.content
    
    # 存储对话到记忆系统（生成回复后）
    if memori:
        store_conversation(memori, user_input, ai_response)
    
    return ai_response


# ============== 设置面板（侧边栏展开器） ==============
def render_settings_panel():
    """在侧边栏渲染设置面板"""
    with st.expander(f"⚙️ {t('settings')}", expanded=False):
        st.markdown(f"**{t('api_config')}**")
        
        api_key = st.text_input(
            t("google_api_key"),
            value=st.session_state.api_key,
            type="password",
            placeholder=t("api_key_placeholder"),
            help=t("api_key_help"),
            key="sidebar_api_key"
        )
        
        model_name = st.text_input(
            t("model_name"),
            value=st.session_state.model_name,
            placeholder="gemini-2.5-flash",
            help=t("model_help"),
            key="sidebar_model"
        )
        
        temperature = st.slider(
            t("temperature"),
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help=t("temperature_help"),
            key="sidebar_temp"
        )
        
        st.markdown("---")
        st.markdown(f"**{t('memory_mode')}**")
        
        memori_mode = st.selectbox(
            t("memory_mode"),
            options=["conscious", "auto", "combined"],
            index=["conscious", "auto", "combined"].index(st.session_state.get("memori_mode", "auto")),
            format_func=lambda x: {
                "conscious": t("memory_mode_conscious"),
                "auto": t("memory_mode_auto"),
                "combined": t("memory_mode_combined")
            }[x],
            help=t("memory_mode_help"),
            key="sidebar_memori_mode"
        )
        
        st.markdown("---")
        st.markdown(f"**{t('persona_title')}**")
        
        persona = st.text_area(
            t("persona_label"),
            value=st.session_state.persona,
            height=100,
            help=t("persona_help"),
            key="sidebar_persona"
        )
        
        st.markdown("---")
        st.markdown(f"**{t('avatar_settings')}**")
        
        # 用户头像上传
        col_upload_user, col_preview_user = st.columns([2, 1])
        with col_upload_user:
            uploaded_user_avatar = st.file_uploader(
                t("user_avatar"),
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                help=t("avatar_help"),
                key="upload_user_avatar"
            )
            
            if uploaded_user_avatar is not None:
                # 处理上传的头像
                avatar_data = crop_circle_image(uploaded_user_avatar.read(), size=64)
                if avatar_data:
                    st.session_state.user_avatar = avatar_data
                    # 立即保存到配置文件
                    save_config()
                    st.success("✅ 用户头像已更新")
                    st.rerun()
            
            # 显示当前用户头像预览
            if st.session_state.get("user_avatar"):
                if st.button(f"❌ {t('remove_avatar')}", key="remove_user_avatar", use_container_width=True):
                    st.session_state.user_avatar = ""
                    save_config()
                    st.rerun()
        
        with col_preview_user:
            if st.session_state.get("user_avatar"):
                st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 80px;">
                    <img src="{st.session_state.user_avatar}" 
                         style="width: 64px; height: 64px; border-radius: 50%; object-fit: cover; border: 2px solid #667eea;" />
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 80px;">
                    <div style="width: 64px; height: 64px; border-radius: 50%; background: #e3f2fd; display: flex; align-items: center; justify-content: center; font-size: 32px;">👤</div>
                </div>
                """, unsafe_allow_html=True)
        
        # 助手头像上传
        col_upload_assistant, col_preview_assistant = st.columns([2, 1])
        with col_upload_assistant:
            uploaded_assistant_avatar = st.file_uploader(
                t("assistant_avatar"),
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                help=t("avatar_help"),
                key="upload_assistant_avatar"
            )
            
            if uploaded_assistant_avatar is not None:
                # 处理上传的头像
                avatar_data = crop_circle_image(uploaded_assistant_avatar.read(), size=64)
                if avatar_data:
                    st.session_state.assistant_avatar = avatar_data
                    # 立即保存到配置文件
                    save_config()
                    st.success("✅ 助手头像已更新")
                    st.rerun()
            
            # 显示当前助手头像预览
            if st.session_state.get("assistant_avatar"):
                if st.button(f"❌ {t('remove_avatar')}", key="remove_assistant_avatar", use_container_width=True):
                    st.session_state.assistant_avatar = ""
                    save_config()
                    st.rerun()
        
        with col_preview_assistant:
            if st.session_state.get("assistant_avatar"):
                st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 80px;">
                    <img src="{st.session_state.assistant_avatar}" 
                         style="width: 64px; height: 64px; border-radius: 50%; object-fit: cover; border: 2px solid #667eea;" />
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 80px;">
                    <div style="width: 64px; height: 64px; border-radius: 50%; background: #f3e5f5; display: flex; align-items: center; justify-content: center; font-size: 32px;">🤖</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 保存按钮
        if st.button(t("save_settings"), type="primary", use_container_width=True, key="save_btn"):
            st.session_state.api_key = api_key
            st.session_state.model_name = model_name
            st.session_state.temperature = temperature
            st.session_state.persona = persona
            st.session_state.memori_mode = memori_mode
            
            # 如果 API key 或记忆模式改变，清除缓存
            if (api_key != st.session_state.get("_last_api_key", "") or 
                memori_mode != st.session_state.get("_last_memori_mode", "")):
                st.cache_resource.clear()
                st.session_state._last_api_key = api_key
                st.session_state._last_memori_mode = memori_mode
            
            if save_config():
                st.success(t("settings_saved"))
                st.rerun()
        
        # 状态指示
        if st.session_state.api_key:
            st.success(t("api_key_configured"))
        else:
            st.warning(t("api_key_required"))


# ============== 记忆面板（侧边栏展开器） ==============
def render_memory_panel():
    """在侧边栏渲染记忆面板"""
    with st.expander(f"🧠 {t('memory_status')}", expanded=False):
        db_file = Path("local_memory.db")
        
        if db_file.exists():
            size_kb = db_file.stat().st_size / 1024
            st.markdown(f"""
            <div class="memory-card">
                <span class="status-indicator status-active"></span>
                <strong>{t('memory_connected')}</strong><br>
                📁 {t('memory_path')}: local_memory.db<br>
                💾 {t('memory_size')}: {size_kb:.1f} KB
            </div>
            """, unsafe_allow_html=True)
            
            memori = get_memori_instance()
            if memori:
                try:
                    # 获取最近的对话历史
                    recent_conversations = memori.get_conversation_history(limit=3)
                    if recent_conversations:
                        st.markdown(f"{t('recent_count')}: **{len(recent_conversations)}**")
                        for i, conv in enumerate(recent_conversations, 1):
                            user_msg = conv.get('user_input', '')
                            if user_msg:
                                display_content = user_msg[:60] + "..." if len(user_msg) > 60 else user_msg
                                st.caption(f"#{i}: {display_content}")
                    else:
                        st.info(t("no_memories"))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    st.info(t("memory_ready"))
            else:
                st.info(t("config_api_first"))
        else:
            st.markdown(f"""
            <div class="memory-card">
                <span class="status-indicator status-inactive"></span>
                <strong>{t('memory_not_created')}</strong><br>
                {t('memory_will_create')}
            </div>
            """, unsafe_allow_html=True)
            st.info(t("memory_tip"))


# ============== 侧边栏 ==============
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        # ===== 设置面板 =====
        render_settings_panel()
        
        # ===== 记忆面板 =====
        render_memory_panel()
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # ===== 新对话按钮 =====
        if st.button(t("new_chat"), use_container_width=True, type="primary"):
            new_id = create_new_conversation()
            st.session_state.current_conversation_id = new_id
            st.session_state.editing_message_index = None
            st.rerun()
        
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f'<p class="sidebar-section-title">{t("chat_history")}</p>', unsafe_allow_html=True)
        
        conversations = list(st.session_state.conversations.values())
        conversations.sort(key=lambda x: x["created_at"], reverse=True)
        
        for conv in conversations:
            conv_id = conv["id"]
            is_active = conv_id == st.session_state.current_conversation_id
            
            col1, col2 = st.columns([5, 1])
            
            with col1:
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    f"{'📌 ' if is_active else '💭 '}{conv['title'][:12]}",
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    type=btn_type
                ):
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.editing_message_index = None
                    save_conversations()  # 保存对话历史
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}", help=t("delete")):
                    if len(st.session_state.conversations) > 1:
                        del st.session_state.conversations[conv_id]
                        if conv_id == st.session_state.current_conversation_id:
                            remaining = list(st.session_state.conversations.keys())
                            st.session_state.current_conversation_id = remaining[0]
                        save_conversations()  # 保存对话历史
                        st.rerun()
            
            st.caption(f"🕐 {conv['created_at']}")


# ============== 顶部导航栏 ==============
def render_topbar():
    """渲染顶部导航栏 - 侧边栏切换按钮和语言切换按钮"""
    lang = st.session_state.get("language", "zh")
    btn_text = "EN" if lang == "zh" else "中"
    btn_tooltip = "Switch to English" if lang == "zh" else "切换到中文"
    
    st.markdown(f"""
    <style>
        .topbar-buttons {{
            position: fixed;
            top: 14px;
            right: 20px;
            z-index: 1000;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .lang-switch-btn {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 6px 16px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            color: #333;
            transition: all 0.2s;
        }}
        .lang-switch-btn:hover {{
            background: #f5f5f5;
            border-color: #667eea;
            color: #667eea;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # 使用 Streamlit 原生按钮（放在右侧列中）
    _, col_lang = st.columns([12, 1])
    with col_lang:
        if st.button(btn_text, key="lang_btn", help=btn_tooltip):
            st.session_state.language = "en" if lang == "zh" else "zh"
            save_config()
            st.rerun()


# ============== 主聊天区 ==============
def render_chat():
    """渲染主聊天界面"""
    # 处理待更新的标题（异步生成完成后的更新）
    if "_pending_title_updates" in st.session_state:
        for conv_id, title in st.session_state["_pending_title_updates"].items():
            if conv_id in st.session_state.conversations:
                st.session_state.conversations[conv_id]["title"] = title
                st.session_state.conversations[conv_id]["title_generated"] = True
        del st.session_state["_pending_title_updates"]
        save_conversations()  # 保存标题更新
    
    render_topbar()
    
    st.markdown(f'<h1 class="main-header">{t("main_title")}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{t("sub_title")}</p>', unsafe_allow_html=True)
    
    messages = get_current_messages()
    last_user_idx = get_last_user_message_index()
    
    messages_container = st.container()
    
    with messages_container:
        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            is_last_user = (role == "user" and idx == last_user_idx)
            
            if role == "user":
                col1, col2, col3 = st.columns([1, 6, 1])
                with col2:
                    if st.session_state.editing_message_index == idx:
                        edit_text = st.text_area(
                            t("edit_message"),
                            value=content,
                            key=f"edit_area_{idx}",
                            height=100,
                            label_visibility="collapsed"
                        )
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button(t("send"), key=f"save_{idx}", use_container_width=True):
                                if edit_text.strip():
                                    update_last_user_message(edit_text.strip())
                                    st.session_state.editing_message_index = None
                                    response = generate_response(edit_text.strip())
                                    add_message_to_current("assistant", response)
                                    conv_id = st.session_state.current_conversation_id
                                    # 优化：异步生成标题，不阻塞响应
                                    if not st.session_state.conversations[conv_id].get("title_generated", False):
                                        def _generate_title_edit():
                                            try:
                                                new_title = generate_conversation_title(get_current_messages())
                                                if "_pending_title_updates" not in st.session_state:
                                                    st.session_state["_pending_title_updates"] = {}
                                                st.session_state["_pending_title_updates"][conv_id] = new_title
                                            except Exception:
                                                pass
                                        thread = threading.Thread(target=_generate_title_edit, daemon=True)
                                        thread.start()
                                        st.session_state.conversations[conv_id]["title_generated"] = True
                                    st.rerun()
                        with col_cancel:
                            if st.button(t("cancel"), key=f"cancel_{idx}", use_container_width=True):
                                st.session_state.editing_message_index = None
                                st.rerun()
                    else:
                        user_avatar_html = get_avatar_html("user", size=36)
                        st.markdown(f"""
                        <div class="message-row user">
                            <div class="message-wrapper">
                                <div class="message-bubble user">{content}</div>
                            </div>
                            <div class="avatar user">{user_avatar_html}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if is_last_user:
                            _, edit_col = st.columns([9.5, 0.5])
                            with edit_col:
                                if st.button("✏️", key=f"edit_btn_{idx}", help=t("edit_resend"), use_container_width=True):
                                    st.session_state.editing_message_index = idx
                                    st.rerun()
            else:
                col1, col2, col3 = st.columns([1, 6, 1])
                with col2:
                    assistant_avatar_html = get_avatar_html("assistant", size=36)
                    st.markdown(f"""
                    <div class="message-row assistant">
                        <div class="avatar assistant">{assistant_avatar_html}</div>
                        <div class="message-wrapper assistant">
                            <div class="message-bubble assistant">{content}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    if prompt := st.chat_input(t("input_placeholder"), key="chat_input"):
        if not st.session_state.api_key:
            st.warning(t("api_key_warning"))
            return
        
        st.session_state.editing_message_index = None
        
        # 清除所有生成相关的标志，确保新消息可以正常处理
        if "_last_processed_user_idx" in st.session_state:
            del st.session_state["_last_processed_user_idx"]
        if "_generating" in st.session_state:
            del st.session_state["_generating"]
        if "_pending_msg" in st.session_state:
            del st.session_state["_pending_msg"]
        
        add_message_to_current("user", prompt)
        st.rerun()
    
    # 检查是否有待生成回复的用户消息
    messages = get_current_messages()
    
    # 简单逻辑：如果最后一条消息是user，就生成回复（因为消息是按顺序添加的）
    if messages and messages[-1]["role"] == "user":
        # 使用消息索引来确保每条消息只处理一次
        last_user_idx = len(messages) - 1
        last_processed_idx = st.session_state.get("_last_processed_user_idx", -1)
        
        # 只有当这条消息还没有被处理过时才生成回复
        if last_user_idx != last_processed_idx:
            # 标记这条消息正在处理
            st.session_state["_last_processed_user_idx"] = last_user_idx
            
            user_msg = messages[-1]["content"]
            
            # 显示思考状态
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                thinking_placeholder = st.empty()
                with thinking_placeholder:
                    st.info(f"🤔 {t('thinking')}...")
            
            # 生成AI回复
            try:
                response = generate_response(user_msg)
                add_message_to_current("assistant", response)
                
                # 清除思考状态
                thinking_placeholder.empty()
                
                conv_id = st.session_state.current_conversation_id
                conv = st.session_state.conversations.get(conv_id, {})
                # 优化：异步生成标题，不阻塞响应
                if not conv.get("title_generated", False) and len(get_current_messages()) >= 2:
                    def _generate_title():
                        """后台生成标题"""
                        try:
                            new_title = generate_conversation_title(get_current_messages())
                            # 使用session_state标记需要更新标题
                            if "_pending_title_updates" not in st.session_state:
                                st.session_state["_pending_title_updates"] = {}
                            st.session_state["_pending_title_updates"][conv_id] = new_title
                        except Exception:
                            pass
                    
                    # 在后台线程执行标题生成
                    thread = threading.Thread(target=_generate_title, daemon=True)
                    thread.start()
                    
                    # 标记标题正在生成，避免重复生成
                    st.session_state.conversations[conv_id]["title_generated"] = True
                
                # 清除处理标记，准备处理下一条消息
                if "_last_processed_user_idx" in st.session_state:
                    del st.session_state["_last_processed_user_idx"]
                st.rerun()
                
            except Exception as e:
                thinking_placeholder.empty()
                if "_last_processed_user_idx" in st.session_state:
                    del st.session_state["_last_processed_user_idx"]
                st.error(f"{t('error_generating')}: {str(e)}")
                st.rerun()


# ============== 主程序入口 ==============
def main():
    """主程序入口"""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
