import os
import io
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import httpx
import html

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters, ConversationHandler
from telegram.error import BadRequest

# Import Google Generative AI
from google import genai

# Import deep_translator for Vietnamese translation
from deep_translator import GoogleTranslator

# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# States for conversation handler
MAIN_MENU, ANALYZING, UPLOADING, LANGUAGE_SELECTION = range(4)

# Direct API keys (since this is a personal project)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Replace with your actual Telegram bot token
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")    # Replace with your actual Google API key

class SimplePDFBot:
    """A user-friendly Telegram bot for PDF analysis using Google's Gemini model."""
    
    def __init__(self):
        """Initialize the bot with hardcoded API keys."""
        # Set Google API key directly
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        
        # Initialize the Google Generative AI client
        self.client = genai.Client()
        
        # Store user data: {user_id: {"files": {file_name: file_ref}, "current_file": file_name, "language": "en|vi", "messages": [message_ids]}}
        self.user_data = {}
        
        # Create the Telegram application
        self.application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        self.add_handlers()
        
        # Unsupported HTML tags that need to be replaced
        self.unsupported_tags = {
            'sub': '_',       # subscript to underscore
            'sup': '^',       # superscript to caret
            'h1': 'b',        # headers to bold
            'h2': 'b',
            'h3': 'b',
            'h4': 'b',
            'h5': 'b',
            'h6': 'b',
            'em': 'i',        # emphasis to italic
            'strong': 'b',    # strong to bold
            'table': None,    # tables and related tags are removed
            'tr': None,
            'td': None,
            'th': None,
            'thead': None,
            'tbody': None,
            'div': None,      # div to nothing (keep content)
            'span': None,     # span to nothing (keep content)
            'p': None,        # paragraph to nothing (keep content)
            'br': '\n',       # line break to newline
            'hr': '\n—————\n', # horizontal rule to dashes
            'img': '[IMAGE]',  # image to text placeholder
            'figure': None,    # figure to nothing
            'figcaption': 'i', # figcaption to italic
        }
    
    def add_handlers(self):
        """Add command and message handlers to the application."""
        # Main conversation handler
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler("start", self.start_command)],
            states={
                MAIN_MENU: [
                    CommandHandler("start", self.start_command),  # Allow restart at any time
                    CommandHandler("menu", self.show_menu),
                    MessageHandler(filters.Document.PDF, self.handle_pdf),
                    CallbackQueryHandler(self.handle_callback_query),  # Handle all callbacks
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_in_menu),
                ],
                ANALYZING: [
                    CommandHandler("start", self.start_command),  # Allow restart at any time
                    CallbackQueryHandler(self.handle_callback_query),  # Handle all callbacks
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_analysis_prompt),
                    CommandHandler("back", self.back_to_menu),
                ],
                UPLOADING: [
                    CommandHandler("start", self.start_command),  # Allow restart at any time
                    MessageHandler(filters.Document.PDF, self.handle_pdf_in_upload),
                    CommandHandler("back", self.back_to_menu),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_in_upload),
                ],
                LANGUAGE_SELECTION: [
                    CommandHandler("start", self.start_command),  # Allow restart at any time
                    CallbackQueryHandler(self.handle_language_selection),
                ],
            },
            fallbacks=[CommandHandler("cancel", self.cancel)],
            per_message=False,  # Set this to False to avoid the warning
            name="main_conversation",
        )
        
        self.application.add_handler(conv_handler)
        
        # Basic commands that work in any state
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("language", self.language_command))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    def sanitize_html(self, text: str) -> str:
        """
        Sanitize HTML content to make it compatible with Telegram's HTML parser.
        Removes or replaces unsupported HTML tags.
        """
        if not text:
            return ""
        
        # First, escape any HTML entities that might be in the text
        text = html.escape(text)
        
        # Then unescape the basic HTML tags that Telegram supports
        for tag in ['b', 'i', 'u', 's', 'code', 'pre', 'a']:
            text = text.replace(f'&lt;{tag}&gt;', f'<{tag}>')
            text = text.replace(f'&lt;/{tag}&gt;', f'</{tag}>')
            # Also handle tags with attributes (simplified approach)
            text = re.sub(f'&lt;{tag}\\s+([^&]*)&gt;', f'<{tag} \\1>', text)
        
        # Handle link tags specially
        text = re.sub(r'&lt;a\s+href=&quot;([^&]*)&quot;&gt;', r'<a href="\1">', text)
        
        # Replace unsupported tags with supported alternatives or remove them
        for tag, replacement in self.unsupported_tags.items():
            if replacement is None:
                # Remove the tag but keep its content
                text = re.sub(f'<{tag}[^>]*>(.*?)</{tag}>', r'\1', text, flags=re.DOTALL)
                text = re.sub(f'<{tag}[^>]*>', '', text)
                text = re.sub(f'</{tag}>', '', text)
            elif replacement == '\n':
                # Replace with newline
                text = re.sub(f'<{tag}[^>]*>', replacement, text)
            else:
                # Replace with another tag
                text = re.sub(f'<{tag}[^>]*>(.*?)</{tag}>', f'<{replacement}>\\1</{replacement}>', text, flags=re.DOTALL)
        
        # Handle special characters and mathematical symbols
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        
        # Replace common mathematical notations
        math_replacements = {
            # Subscripts
            '<sub>': '_',
            '</sub>': '',
            # Superscripts
            '<sup>': '^',
            '</sup>': '',
            # Greek letters (common ones)
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'θ': 'theta',
            'λ': 'lambda',
            'μ': 'mu',
            'π': 'pi',
            'σ': 'sigma',
            'τ': 'tau',
            'φ': 'phi',
            'ω': 'omega',
            # Math symbols
            '±': '+/-',
            '×': 'x',
            '÷': '/',
            '≈': '~=',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '∞': 'inf',
            '∑': 'sum',
            '∏': 'prod',
            '∫': 'int',
            '∂': 'partial',
            '√': 'sqrt',
            '∛': 'cbrt',
            '∝': 'prop to',
        }
        
        for symbol, replacement in math_replacements.items():
            text = text.replace(symbol, replacement)
        
        # Handle tables by converting them to plain text format
        # This is a simplified approach - tables will lose their structure
        if '<table' in text:
            # Extract table content and format it as plain text
            table_pattern = r'<table[^>]*>(.*?)</table>'
            for table_match in re.finditer(table_pattern, text, re.DOTALL):
                table_content = table_match.group(1)
                # Replace table rows with newlines
                table_content = re.sub(r'<tr[^>]*>(.*?)</tr>', r'\1\n', table_content, flags=re.DOTALL)
                # Replace table cells with tab-separated text
                table_content = re.sub(r'<t[hd][^>]*>(.*?)</t[hd]>', r'\1\t', table_content, flags=re.DOTALL)
                # Replace the table with the formatted content
                text = text.replace(table_match.group(0), f"\n{table_content}\n")
        
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix common issues with Telegram's HTML parser
        # Double newlines for paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Ensure we don't have more than 2 consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    async def send_safe_message(self, chat_id: int, text: str, reply_markup=None, parse_mode=ParseMode.HTML) -> Any:
        """
        Send a message with safe HTML parsing, handling any formatting errors.
        Falls back to plain text if HTML parsing fails.
        """
        try:
            # First try with the original text and HTML parsing
            return await self.application.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode
            )
        except BadRequest as e:
            logger.warning(f"HTML parsing error: {e}")
            
            if "Can't parse entities" in str(e):
                # Try with sanitized HTML
                sanitized_text = self.sanitize_html(text)
                try:
                    return await self.application.bot.send_message(
                        chat_id=chat_id,
                        text=sanitized_text,
                        reply_markup=reply_markup,
                        parse_mode=parse_mode
                    )
                except BadRequest as e2:
                    logger.warning(f"Still can't parse entities after sanitization: {e2}")
                    
                    # As a last resort, send without parse_mode
                    return await self.application.bot.send_message(
                        chat_id=chat_id,
                        text=sanitized_text,
                        reply_markup=reply_markup,
                        parse_mode=None
                    )
            else:
                # For other types of errors, re-raise
                raise
    
    async def edit_safe_message(self, chat_id: int, message_id: int, text: str, reply_markup=None, parse_mode=ParseMode.HTML) -> Any:
        """
        Edit a message with safe HTML parsing, handling any formatting errors.
        Falls back to plain text if HTML parsing fails.
        """
        try:
            # First try with the original text and HTML parsing
            return await self.application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode
            )
        except BadRequest as e:
            logger.warning(f"HTML parsing error in edit: {e}")
            
            if "Can't parse entities" in str(e):
                # Try with sanitized HTML
                sanitized_text = self.sanitize_html(text)
                try:
                    return await self.application.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=sanitized_text,
                        reply_markup=reply_markup,
                        parse_mode=parse_mode
                    )
                except BadRequest as e2:
                    logger.warning(f"Still can't parse entities after sanitization: {e2}")
                    
                    # As a last resort, send without parse_mode
                    return await self.application.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=sanitized_text,
                        reply_markup=reply_markup,
                        parse_mode=None
                    )
            elif "Message is not modified" in str(e):
                # Message content is the same, ignore this error
                logger.info("Message not modified, ignoring")
                return None
            else:
                # For other types of errors, re-raise
                raise
    
    async def split_and_send_long_message(self, update: Update, text: str, title: str = "", reply_markup=None) -> List[int]:
        """
        Split long messages and send them in chunks.
        Returns a list of message IDs that were sent.
        """
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        message_ids = []
        
        # Maximum message length for Telegram
        max_length = 4000
        
        # If the message is short enough, send it as is
        if len(text) <= max_length:
            message = await self.send_safe_message(
                chat_id=chat_id,
                text=f"{title}\n\n{text}" if title else text,
                reply_markup=reply_markup
            )
            message_ids.append(message.message_id)
            self.add_message_to_cleanup(user_id, message.message_id)
            return message_ids
        
        # Split by paragraphs first to maintain readability
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the limit, start a new chunk
            if len(current_chunk) + len(paragraph) + 2 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += '\n\n'
                current_chunk += paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # If we still have chunks that are too long, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Split by sentences or just characters if needed
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                sub_chunk = ""
                for sentence in sentences:
                    if len(sub_chunk) + len(sentence) + 1 > max_length:
                        if sub_chunk:
                            final_chunks.append(sub_chunk)
                        # If a single sentence is too long, split it by characters
                        if len(sentence) > max_length:
                            for i in range(0, len(sentence), max_length):
                                final_chunks.append(sentence[i:i+max_length])
                        else:
                            sub_chunk = sentence
                    else:
                        if sub_chunk:
                            sub_chunk += ' '
                        sub_chunk += sentence
                if sub_chunk:
                    final_chunks.append(sub_chunk)
        
        # Send each chunk
        for i, chunk in enumerate(final_chunks):
            part_title = f"{title} (phần {i+1}/{len(final_chunks)})" if title else f"Phần {i+1}/{len(final_chunks)}"
            
            # Only add reply_markup to the last chunk
            current_markup = reply_markup if i == len(final_chunks) - 1 else None
            
            message = await self.send_safe_message(
                chat_id=chat_id,
                text=f"{part_title}\n\n{chunk}",
                reply_markup=current_markup
            )
            message_ids.append(message.message_id)
            self.add_message_to_cleanup(user_id, message.message_id)
        
        return message_ids
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """General handler for all callback queries."""
        query = update.callback_query
        data = query.data
        
        # Log the callback data for debugging
        logger.info(f"Received callback query with data: {data}")
        
        try:
            # Route to appropriate handler based on prefix
            if data.startswith("analyze_"):
                return await self.handle_analysis_callback(update, context)
            elif data.startswith("menu_"):
                return await self.handle_menu_callback(update, context)
            elif data.startswith("compare_"):
                return await self.handle_menu_callback(update, context)
            elif data.startswith("lang_"):
                return await self.handle_language_selection(update, context)
            else:
                # Default fallback
                await query.answer("Unknown button")
                return MAIN_MENU
        except Exception as e:
            logger.error(f"Error handling callback query: {e}")
            await query.answer(f"Error: {str(e)[:200]}")
            return MAIN_MENU
    
    async def language_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle the /language command to change the language."""
        user_id = update.effective_user.id
        
        # Initialize user data if not exists
        if user_id not in self.user_data:
            self.user_data[user_id] = {"files": {}, "current_file": None, "language": "en", "messages": []}
        
        keyboard = [
            [InlineKeyboardButton("🇬🇧 English", callback_data="lang_en")],
            [InlineKeyboardButton("🇻🇳 Tiếng Việt", callback_data="lang_vi")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = await update.message.reply_text(
            "🌐 Please select your preferred language / Vui lòng chọn ngôn ngữ:",
            reply_markup=reply_markup
        )
        
        # Track this message for cleanup
        self.add_message_to_cleanup(user_id, message.message_id)
        
        return LANGUAGE_SELECTION
    
    async def handle_language_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle language selection callback."""
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        data = query.data
        
        # Initialize user data if not exists
        if user_id not in self.user_data:
            self.user_data[user_id] = {"files": {}, "current_file": None, "language": "en", "messages": []}
        
        if data == "lang_en":
            self.user_data[user_id]["language"] = "en"
            await query.edit_message_text("✅ Language set to English")
        elif data == "lang_vi":
            self.user_data[user_id]["language"] = "vi"
            await query.edit_message_text("✅ Đã chọn Tiếng Việt")
        
        # Return to main menu after a short delay
        await asyncio.sleep(1)
        return await self.show_menu(update, context)
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to the target language."""
        if target_language == "en" or not text:
            return text
        
        try:
            # Split text into manageable chunks (Google Translator has a limit)
            max_chunk_size = 4500
            chunks = []
            
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += '\n\n'
                    current_chunk += paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Translate each chunk
            translated_chunks = []
            for chunk in chunks:
                translated = GoogleTranslator(source='auto', target=target_language).translate(chunk)
                translated_chunks.append(translated)
            
            return '\n\n'.join(translated_chunks)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text + "\n\n(Translation failed)"
    
    def add_message_to_cleanup(self, user_id: int, message_id: int) -> None:
        """Add a message ID to the user's cleanup list."""
        if user_id not in self.user_data:
            self.user_data[user_id] = {"files": {}, "current_file": None, "language": "en", "messages": []}
        
        if "messages" not in self.user_data[user_id]:
            self.user_data[user_id]["messages"] = []
        
        self.user_data[user_id]["messages"].append(message_id)
    
    async def cleanup_messages(self, update: Update, max_keep: int = 3) -> None:
        """Clean up old messages to avoid cluttering the chat."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        if user_id not in self.user_data or "messages" not in self.user_data[user_id]:
            return
        
        # Keep only the most recent messages
        messages_to_delete = self.user_data[user_id]["messages"][:-max_keep] if len(self.user_data[user_id]["messages"]) > max_keep else []
        
        for msg_id in messages_to_delete:
            try:
                await update.get_bot().delete_message(chat_id=chat_id, message_id=msg_id)
            except BadRequest as e:
                # Message may already be deleted or too old
                logger.info(f"Could not delete message {msg_id}: {e}")
        
        # Update the messages list
        self.user_data[user_id]["messages"] = self.user_data[user_id]["messages"][-max_keep:] if len(self.user_data[user_id]["messages"]) > max_keep else self.user_data[user_id]["messages"]
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start the conversation and show the main menu."""
        user = update.effective_user
        user_id = user.id
        
        # Initialize user data if not exists
        if user_id not in self.user_data:
            self.user_data[user_id] = {"files": {}, "current_file": None, "language": "en", "messages": []}
        
        # Clean up old messages
        await self.cleanup_messages(update)
        
        # Determine language
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        welcome_text_en = (
            f"👋 Hi {user.mention_html()}! I'm your PDF Analysis Assistant.\n\n"
            f"I can help you analyze PDF documents using Google's Gemini AI.\n\n"
            f"🔍 <b>What I can do:</b>\n"
            f"• Analyze PDF documents\n"
            f"• Extract key information\n"
            f"• Answer questions about your documents\n"
            f"• Compare multiple documents\n\n"
            f"Let's get started!"
        )
        
        welcome_text_vi = (
            f"👋 Chào {user.mention_html()}! Tôi là Trợ lý Phân tích PDF của bạn.\n\n"
            f"Tôi có thể giúp bạn phân tích tài liệu PDF bằng AI Gemini của Google.\n\n"
            f"🔍 <b>Tôi có thể làm gì:</b>\n"
            f"• Phân tích tài liệu PDF\n"
            f"• Trích xuất thông tin quan trọng\n"
            f"• Trả lời câu hỏi về tài liệu của bạn\n"
            f"• So sánh nhiều tài liệu\n\n"
            f"Hãy bắt đầu!"
        )
        
        welcome_text = welcome_text_vi if is_vietnamese else welcome_text_en
        
        message = await update.message.reply_html(welcome_text)
        self.add_message_to_cleanup(user_id, message.message_id)
        
        return await self.show_menu(update, context)
    
    async def show_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Show the main menu with options."""
        user_id = update.effective_user.id
        
        # Clean up old messages
        await self.cleanup_messages(update)
        
        # Check if user has any files
        has_files = bool(self.user_data[user_id].get("files"))
        current_file = self.user_data[user_id].get("current_file")
        
        # Determine language
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        # Create menu buttons
        if is_vietnamese:
            keyboard = [
                [InlineKeyboardButton("📤 Tải lên PDF", callback_data="menu_upload")],
            ]
            
            if has_files:
                keyboard.append([InlineKeyboardButton("📚 Tài liệu của tôi", callback_data="menu_files")])
                
                if current_file:
                    keyboard.append([InlineKeyboardButton(f"📝 Phân tích: {current_file}", callback_data="menu_analyze")])
                    keyboard.append([InlineKeyboardButton("❓ Đặt câu hỏi", callback_data="menu_ask")])
                
                if len(self.user_data[user_id]["files"]) >= 2:
                    keyboard.append([InlineKeyboardButton("🔄 So sánh tài liệu", callback_data="menu_compare")])
            
            keyboard.append([InlineKeyboardButton("🌐 Ngôn ngữ / Language", callback_data="menu_language")])
        else:
            keyboard = [
                [InlineKeyboardButton("📤 Upload PDF", callback_data="menu_upload")],
            ]
            
            if has_files:
                keyboard.append([InlineKeyboardButton("📚 My Documents", callback_data="menu_files")])
                
                if current_file:
                    keyboard.append([InlineKeyboardButton(f"📝 Analyze: {current_file}", callback_data="menu_analyze")])
                    keyboard.append([InlineKeyboardButton("❓ Ask Question", callback_data="menu_ask")])
                
                if len(self.user_data[user_id]["files"]) >= 2:
                    keyboard.append([InlineKeyboardButton("🔄 Compare Documents", callback_data="menu_compare")])
            
            keyboard.append([InlineKeyboardButton("🌐 Language / Ngôn ngữ", callback_data="menu_language")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Menu text based on language
        if is_vietnamese:
            menu_title = "📋 <b>Menu Chính</b>"
            current_doc_text = f"✅ Tài liệu hiện tại: {current_file}" if current_file else "❗ Chưa chọn tài liệu"
            choose_option = "Chọn một tùy chọn:"
        else:
            menu_title = "📋 <b>Main Menu</b>"
            current_doc_text = f"✅ Current document: {current_file}" if current_file else "❗ No document selected"
            choose_option = "Choose an option:"
        
        menu_text = f"{menu_title}\n\n{current_doc_text}\n\n{choose_option}"
        
        # Determine if this is a new message or an edit
        if update.callback_query:
            try:
                await update.callback_query.answer()
                message = await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=update.callback_query.message.message_id,
                    text=menu_text,
                    reply_markup=reply_markup
                )
                # No need to track edited messages
            except BadRequest as e:
                # If we can't edit (e.g., message is too old), send a new one
                logger.info(f"Could not edit message: {e}")
                message = await self.send_safe_message(
                    chat_id=update.effective_chat.id,
                    text=menu_text,
                    reply_markup=reply_markup
                )
                self.add_message_to_cleanup(user_id, message.message_id)
        else:
            message = await self.send_safe_message(
                chat_id=update.effective_chat.id,
                text=menu_text,
                reply_markup=reply_markup
            )
            self.add_message_to_cleanup(user_id, message.message_id)
        
        return MAIN_MENU
    
    async def handle_menu_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle callbacks from the main menu."""
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        data = query.data
        
        # Log the callback data for debugging
        logger.info(f"Menu callback: {data}")
        
        # Determine language
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        if data == "menu_upload":
            upload_text = (
                "📤 <b>Tải lên PDF</b>\n\n"
                "Vui lòng gửi cho tôi một tài liệu PDF để phân tích.\n"
                "Bạn có thể đính kèm tệp PDF.\n\n"
                "Gõ /back để quay lại menu chính."
            ) if is_vietnamese else (
                "📤 <b>Upload PDF</b>\n\n"
                "Please send me a PDF document to analyze.\n"
                "You can simply attach a PDF file.\n\n"
                "Type /back to return to the main menu."
            )
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=upload_text
            )
            return UPLOADING
        
        elif data == "menu_language":
            keyboard = [
                [InlineKeyboardButton("🇬🇧 English", callback_data="lang_en")],
                [InlineKeyboardButton("🇻🇳 Tiếng Việt", callback_data="lang_vi")],
                [InlineKeyboardButton("🔙 Back / Quay lại", callback_data="menu_back")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text="🌐 Please select your language / Vui lòng chọn ngôn ngữ:",
                reply_markup=reply_markup
            )
            return LANGUAGE_SELECTION
        
        elif data == "menu_files":
            # Show list of files with selection buttons
            files = list(self.user_data[user_id]["files"].keys())
            current_file = self.user_data[user_id].get("current_file")
            
            message = "📚 <b>Your Documents</b>\n\n" if not is_vietnamese else "📚 <b>Tài liệu của bạn</b>\n\n"
            
            keyboard = []
            for file in files:
                file_text = f"{file} {'✓' if file == current_file else ''}"
                select_text = f"Chọn: {file_text}" if is_vietnamese else f"Select: {file_text}"
                delete_text = f"❌ Xóa: {file}" if is_vietnamese else f"❌ Delete: {file}"
                
                # Ensure callback data is valid (max 64 bytes)
                select_callback = f"menu_select_{self._safe_callback_data(file)}"
                delete_callback = f"menu_delete_{self._safe_callback_data(file)}"
                
                keyboard.append([InlineKeyboardButton(select_text, callback_data=select_callback)])
                keyboard.append([InlineKeyboardButton(delete_text, callback_data=delete_callback)])
            
            back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
            keyboard.append([InlineKeyboardButton(back_text, callback_data="menu_back")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=message,
                reply_markup=reply_markup
            )
            return MAIN_MENU
        
        elif data == "menu_analyze":
            if not self.user_data[user_id].get("current_file"):
                no_doc_text = (
                    "❗ Chưa chọn tài liệu. Vui lòng tải lên hoặc chọn một tài liệu trước."
                ) if is_vietnamese else (
                    "❗ No document selected. Please upload or select a document first."
                )
                
                back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=no_doc_text,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(back_text, callback_data="menu_back")
                    ]])
                )
                return MAIN_MENU
            
            current_file = self.user_data[user_id]["current_file"]
            
            # Show analysis options
            if is_vietnamese:
                keyboard = [
                    [InlineKeyboardButton("📝 Tóm tắt", callback_data="analyze_summarize")],
                    [InlineKeyboardButton("🔑 Điểm chính", callback_data="analyze_key_points")],
                    [InlineKeyboardButton("📊 Lập luận chính", callback_data="analyze_arguments")],
                    [InlineKeyboardButton("📈 Dữ liệu & Thống kê", callback_data="analyze_data")],
                    [InlineKeyboardButton("🔙 Quay lại Menu", callback_data="analyze_back")]
                ]
                
                analyze_text = f"📝 <b>Phân tích tài liệu</b>: {current_file}\n\n" \
                               f"Chọn loại phân tích hoặc nhập yêu cầu của bạn:"
            else:
                keyboard = [
                    [InlineKeyboardButton("📝 Summarize", callback_data="analyze_summarize")],
                    [InlineKeyboardButton("🔑 Key Points", callback_data="analyze_key_points")],
                    [InlineKeyboardButton("📊 Main Arguments", callback_data="analyze_arguments")],
                    [InlineKeyboardButton("📈 Data & Statistics", callback_data="analyze_data")],
                    [InlineKeyboardButton("🔙 Back to Menu", callback_data="analyze_back")]
                ]
                
                analyze_text = f"📝 <b>Analyze Document</b>: {current_file}\n\n" \
                               f"Choose an analysis type or type your own prompt:"
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=analyze_text,
                reply_markup=reply_markup
            )
            
            return ANALYZING
        
        elif data == "menu_ask":
            if not self.user_data[user_id].get("current_file"):
                no_doc_text = (
                    "❗ Chưa chọn tài liệu. Vui lòng tải lên hoặc chọn một tài liệu trước."
                ) if is_vietnamese else (
                    "❗ No document selected. Please upload or select a document first."
                )
                
                back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=no_doc_text,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(back_text, callback_data="menu_back")
                    ]])
                )
                return MAIN_MENU
            
            current_file = self.user_data[user_id]["current_file"]
            
            if is_vietnamese:
                ask_text = (
                    f"❓ <b>Hỏi về</b>: {current_file}\n\n"
                    f"Nhập câu hỏi của bạn về tài liệu này.\n\n"
                    f"Ví dụ:\n"
                    f"• Kết luận chính là gì?\n"
                    f"• Những bên liên quan chính được đề cập là ai?\n"
                    f"• Phương pháp nào đã được sử dụng?\n\n"
                    f"Gõ /back để quay lại menu chính."
                )
            else:
                ask_text = (
                    f"❓ <b>Ask about</b>: {current_file}\n\n"
                    f"Type your question about this document.\n\n"
                    f"Examples:\n"
                    f"• What is the main conclusion?\n"
                    f"• Who are the key stakeholders mentioned?\n"
                    f"• What methodology was used?\n\n"
                    f"Type /back to return to the main menu."
                )
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=ask_text
            )
            
            return ANALYZING
        
        elif data == "menu_compare":
            files = list(self.user_data[user_id]["files"].keys())
            
            if len(files) < 2:
                need_more_docs = (
                    "❗ Bạn cần ít nhất 2 tài liệu để so sánh. Vui lòng tải lên thêm tài liệu."
                ) if is_vietnamese else (
                    "❗ You need at least 2 documents to compare. Please upload more documents."
                )
                
                back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=need_more_docs,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(back_text, callback_data="menu_back")
                    ]])
                )
                return MAIN_MENU
            
            # Create checkboxes for file selection
            keyboard = []
            for file in files:
                file_display = f"☐ {file}"
                # Ensure callback data is valid (max 64 bytes)
                callback_data = f"compare_select_{self._safe_callback_data(file)}"
                keyboard.append([InlineKeyboardButton(file_display, callback_data=callback_data)])
            
            if is_vietnamese:
                compare_button = "🔄 So sánh đã chọn"
                back_button = "🔙 Quay lại Menu"
                compare_text = (
                    "🔄 <b>So sánh tài liệu</b>\n\n"
                    "Chọn ít nhất 2 tài liệu để so sánh, sau đó nhấp vào 'So sánh đã chọn':"
                )
            else:
                compare_button = "🔄 Compare Selected"
                back_button = "🔙 Back to Menu"
                compare_text = (
                    "🔄 <b>Compare Documents</b>\n\n"
                    "Select at least 2 documents to compare, then click 'Compare Selected':"
                )
            
            keyboard.append([InlineKeyboardButton(compare_button, callback_data="compare_execute")])
            keyboard.append([InlineKeyboardButton(back_button, callback_data="menu_back")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Store selected files in user_data
            self.user_data[user_id]["compare_selection"] = []
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=compare_text,
                reply_markup=reply_markup
            )
            
            return MAIN_MENU
        
        elif data == "menu_back" or data == "analyze_back":
            return await self.show_menu(update, context)
        
        elif data.startswith("menu_select_"):
            # Extract the file name safely
            file_id = data[12:]
            file_name = self._get_filename_from_id(user_id, file_id)
            
            if file_name and file_name in self.user_data[user_id]["files"]:
                self.user_data[user_id]["current_file"] = file_name
                
                selected_text = f"✅ Đã chọn tài liệu: {file_name}" if is_vietnamese else f"✅ Selected document: {file_name}"
                back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=selected_text,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(back_text, callback_data="menu_back")
                    ]])
                )
            return MAIN_MENU
        
        elif data.startswith("menu_delete_"):
            # Extract the file name safely
            file_id = data[12:]
            file_name = self._get_filename_from_id(user_id, file_id)
            
            if file_name and file_name in self.user_data[user_id]["files"]:
                # Delete the file from Google AI
                file_ref = self.user_data[user_id]["files"][file_name]
                try:
                    self.client.files.delete(file_ref.name)
                except Exception as e:
                    logger.error(f"Error deleting file from Google AI: {e}")
                
                # Remove from user data
                del self.user_data[user_id]["files"][file_name]
                
                # Reset current file if it was the deleted one
                if self.user_data[user_id].get("current_file") == file_name:
                    self.user_data[user_id]["current_file"] = None
                
                deleted_text = f"✅ Đã xóa tài liệu: {file_name}" if is_vietnamese else f"✅ Deleted document: {file_name}"
                back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=deleted_text,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(back_text, callback_data="menu_back")
                    ]])
                )
            return MAIN_MENU
        
        elif data.startswith("compare_select_"):
            # Extract the file name safely
            file_id = data[14:]
            file_name = self._get_filename_from_id(user_id, file_id)
            
            if file_name and file_name in self.user_data[user_id]["files"]:
                # Toggle selection
                if file_name in self.user_data[user_id].get("compare_selection", []):
                    self.user_data[user_id]["compare_selection"].remove(file_name)
                else:
                    if "compare_selection" not in self.user_data[user_id]:
                        self.user_data[user_id]["compare_selection"] = []
                    self.user_data[user_id]["compare_selection"].append(file_name)
                
                # Rebuild keyboard with updated selections
                files = list(self.user_data[user_id]["files"].keys())
                keyboard = []
                for file in files:
                    is_selected = file in self.user_data[user_id].get("compare_selection", [])
                    checkbox = "☑" if is_selected else "☐"
                    # Ensure callback data is valid
                    callback_data = f"compare_select_{self._safe_callback_data(file)}"
                    keyboard.append([
                        InlineKeyboardButton(f"{checkbox} {file}", callback_data=callback_data)
                    ])
                
                if is_vietnamese:
                    compare_button = "🔄 So sánh đã chọn"
                    back_button = "🔙 Quay lại Menu"
                    compare_text = (
                        "🔄 <b>So sánh tài liệu</b>\n\n"
                        f"Đã chọn: {len(self.user_data[user_id].get('compare_selection', []))}/2 tài liệu\n\n"
                        "Chọn ít nhất 2 tài liệu để so sánh, sau đó nhấp vào 'So sánh đã chọn':"
                    )
                else:
                    compare_button = "🔄 Compare Selected"
                    back_button = "🔙 Back to Menu"
                    compare_text = (
                        "🔄 <b>Compare Documents</b>\n\n"
                        f"Selected: {len(self.user_data[user_id].get('compare_selection', []))}/2 documents\n\n"
                        "Select at least 2 documents to compare, then click 'Compare Selected':"
                    )
                
                keyboard.append([InlineKeyboardButton(compare_button, callback_data="compare_execute")])
                keyboard.append([InlineKeyboardButton(back_button, callback_data="menu_back")])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=compare_text,
                    reply_markup=reply_markup
                )
            return MAIN_MENU
        
        elif data == "compare_execute":
            selected_files = self.user_data[user_id].get("compare_selection", [])
            
            if len(selected_files) < 2:
                not_enough_text = "❗ Vui lòng chọn ít nhất 2 tài liệu để so sánh." if is_vietnamese else "❗ Please select at least 2 documents to compare."
                back_text = "🔙 Quay lại" if is_vietnamese else "🔙 Back"
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=not_enough_text,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(back_text, callback_data="menu_compare")
                    ]])
                )
                return MAIN_MENU
            
            # Get file references
            file_refs = [self.user_data[user_id]["files"][name] for name in selected_files]
            
            # Show comparison options
            if is_vietnamese:
                keyboard = [
                    [InlineKeyboardButton("📊 So sánh tổng quát", callback_data="compare_general")],
                    [InlineKeyboardButton("🔍 Điểm khác biệt", callback_data="compare_differences")],
                    [InlineKeyboardButton("🔗 Chủ đề chung", callback_data="compare_common")],
                    [InlineKeyboardButton("📈 So sánh dữ liệu", callback_data="compare_data")],
                    [InlineKeyboardButton("🔙 Quay lại", callback_data="menu_compare")]
                ]
                
                compare_text = (
                    f"🔄 <b>So sánh tài liệu</b>\n\n"
                    f"Tài liệu đã chọn:\n"
                    f"• {selected_files[0]}\n"
                    f"• {selected_files[1]}\n"
                    f"{f'• {selected_files[2]}' if len(selected_files) > 2 else ''}\n\n"
                    f"Chọn kiểu so sánh:"
                )
            else:
                keyboard = [
                    [InlineKeyboardButton("📊 General Comparison", callback_data="compare_general")],
                    [InlineKeyboardButton("🔍 Key Differences", callback_data="compare_differences")],
                    [InlineKeyboardButton("🔗 Common Themes", callback_data="compare_common")],
                    [InlineKeyboardButton("📈 Data Comparison", callback_data="compare_data")],
                    [InlineKeyboardButton("🔙 Back", callback_data="menu_compare")]
                ]
                
                compare_text = (
                    f"🔄 <b>Compare Documents</b>\n\n"
                    f"Selected documents:\n"
                    f"• {selected_files[0]}\n"
                    f"• {selected_files[1]}\n"
                    f"{f'• {selected_files[2]}' if len(selected_files) > 2 else ''}\n\n"
                    f"Choose comparison type:"
                )
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=compare_text,
                reply_markup=reply_markup
            )
            
            return MAIN_MENU
        
        elif data.startswith("compare_"):
            comparison_type = data[8:]
            selected_files = self.user_data[user_id].get("compare_selection", [])
            
            if len(selected_files) < 2:
                not_enough_text = "❗ Vui lòng chọn ít nhất 2 tài liệu để so sánh." if is_vietnamese else "❗ Please select at least 2 documents to compare."
                back_text = "🔙 Quay lại" if is_vietnamese else "🔙 Back"
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=not_enough_text,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(back_text, callback_data="menu_compare")
                    ]])
                )
                return MAIN_MENU
            
            # Get file references
            file_refs = [self.user_data[user_id]["files"][name] for name in selected_files]
            
            # Different prompts based on comparison type
            if is_vietnamese:
                prompts = {
                    "general": "So sánh các tài liệu này và cung cấp tổng quan về điểm giống và khác nhau.",
                    "differences": "Những điểm khác biệt chính giữa các tài liệu này là gì? Tập trung vào quan điểm, phương pháp, hoặc kết luận trái ngược.",
                    "common": "Xác định và giải thích các chủ đề, lập luận, hoặc phát hiện chung trong các tài liệu này.",
                    "data": "So sánh bất kỳ dữ liệu, thống kê, hoặc thông tin số nào được trình bày trong các tài liệu này. Tạo bảng nếu thích hợp."
                }
            else:
                prompts = {
                    "general": "Compare these documents and provide a general overview of their similarities and differences.",
                    "differences": "What are the key differences between these documents? Focus on contrasting viewpoints, methodologies, or conclusions.",
                    "common": "Identify and explain the common themes, arguments, or findings shared across these documents.",
                    "data": "Compare any data, statistics, or numerical information presented in these documents. Create a table if appropriate."
                }
            
            prompt = prompts.get(comparison_type, prompts["general"])
            
            # Indicate processing
            processing_text = "⏳ Đang so sánh tài liệu...\n\nQuá trình này có thể mất một phút tùy thuộc vào kích thước và độ phức tạp của tài liệu." if is_vietnamese else "⏳ Comparing documents...\n\nThis may take a minute depending on the size and complexity of your documents."
            
            processing_message = await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=processing_text
            )
            
            # Compare the documents
            try:
                response = await self.compare_documents(file_refs, prompt)
                
                # Translate if needed
                if is_vietnamese:
                    response = await self.translate_text(response, "vi")
                
                # Delete the processing message
                try:
                    await self.application.bot.delete_message(
                        chat_id=update.effective_chat.id,
                        message_id=query.message.message_id
                    )
                except Exception as e:
                    logger.info(f"Could not delete processing message: {e}")
                
                # Send results as a new message
                result_title = "📊 <b>Kết quả so sánh</b>" if is_vietnamese else "📊 <b>Comparison Results</b>"
                
                # Use the new split and send method
                await self.split_and_send_long_message(
                    update=update,
                    text=response,
                    title=result_title
                )
                
                # Return to menu
                return await self.show_menu(update, context)
            except Exception as e:
                logger.error(f"Error comparing documents: {e}")
                
                error_text = f"❌ Lỗi khi so sánh tài liệu: {str(e)}\n\nVui lòng thử lại." if is_vietnamese else f"❌ Error comparing documents: {str(e)}\n\nPlease try again."
                back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
                
                await self.edit_safe_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id,
                    text=error_text,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(back_text, callback_data="menu_back")
                    ]])
                )
                
                return MAIN_MENU
        
        return MAIN_MENU
    
    async def handle_analysis_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle callbacks from the analysis menu."""
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        data = query.data
        
        # Log the callback data for debugging
        logger.info(f"Analysis callback: {data}")
        
        # Determine language
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        if data == "analyze_back":
            return await self.show_menu(update, context)
        
        if not self.user_data[user_id].get("current_file"):
            no_doc_text = "❗ Chưa chọn tài liệu. Vui lòng tải lên hoặc chọn một tài liệu trước." if is_vietnamese else "❗ No document selected. Please upload or select a document first."
            back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=no_doc_text,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(back_text, callback_data="menu_back")
                ]])
            )
            return MAIN_MENU
        
        file_name = self.user_data[user_id]["current_file"]
        file_ref = self.user_data[user_id]["files"][file_name]
        
        # Different prompts based on analysis type
        if is_vietnamese:
            prompts = {
                "analyze_summarize": "Tóm tắt tài liệu này một cách ngắn gọn, nhấn mạnh thông tin quan trọng nhất.",
                "analyze_key_points": "Trích xuất và liệt kê các điểm chính từ tài liệu này.",
                "analyze_arguments": "Những lập luận hoặc tuyên bố chính được trình bày trong tài liệu này là gì?",
                "analyze_data": "Trích xuất và tổ chức bất kỳ dữ liệu, thống kê, hoặc thông tin số nào từ tài liệu này."
            }
        else:
            prompts = {
                "analyze_summarize": "Summarize this document in a concise way, highlighting the most important information.",
                "analyze_key_points": "Extract and list the key points from this document.",
                "analyze_arguments": "What are the main arguments or claims presented in this document?",
                "analyze_data": "Extract and organize any data, statistics, or numerical information from this document."
            }
        
        prompt = prompts.get(data, prompts["analyze_summarize"])
        
        # Indicate processing
        processing_text = f"⏳ Đang phân tích tài liệu: {file_name}...\n\nQuá trình này có thể mất một phút tùy thuộc vào kích thước và độ phức tạp của tài liệu." if is_vietnamese else f"⏳ Analyzing document: {file_name}...\n\nThis may take a minute depending on the size and complexity of your document."
        
        processing_message = await self.edit_safe_message(
            chat_id=update.effective_chat.id,
            message_id=query.message.message_id,
            text=processing_text
        )
        
        # Analyze the document
        try:
            response = await self.analyze_document(file_ref, prompt)
            
            # Translate if needed
            if is_vietnamese:
                response = await self.translate_text(response, "vi")
            
            # Delete the processing message
            try:
                await self.application.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=query.message.message_id
                )
            except Exception as e:
                logger.info(f"Could not delete processing message: {e}")
            
            # Send results as a new message
            result_title = "📝 <b>Kết quả phân tích</b>" if is_vietnamese else "📝 <b>Analysis Results</b>"
            
            # Use the new split and send method
            await self.split_and_send_long_message(
                update=update,
                text=response,
                title=result_title
            )
            
            # Return to menu with follow-up options
            if is_vietnamese:
                follow_up_text = "Bạn muốn làm gì tiếp theo?"
                ask_button = "🔍 Đặt câu hỏi tiếp theo"
                back_button = "🔙 Quay lại Menu"
            else:
                follow_up_text = "What would you like to do next?"
                ask_button = "🔍 Ask Follow-up Question"
                back_button = "🔙 Back to Menu"
            
            keyboard = [
                [InlineKeyboardButton(ask_button, callback_data="menu_ask")],
                [InlineKeyboardButton(back_button, callback_data="menu_back")]
            ]
            
            message = await self.send_safe_message(
                chat_id=update.effective_chat.id,
                text=follow_up_text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            self.add_message_to_cleanup(user_id, message.message_id)
            
            return MAIN_MENU
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            
            error_text = f"❌ Lỗi khi phân tích tài liệu: {str(e)}\n\nVui lòng thử lại." if is_vietnamese else f"❌ Error analyzing document: {str(e)}\n\nPlease try again."
            back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
            
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                text=error_text,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(back_text, callback_data="menu_back")
                ]])
            )
            
            return MAIN_MENU
    
    async def handle_analysis_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle custom analysis prompts."""
        user_id = update.effective_user.id
        
        # Determine language
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        if not self.user_data[user_id].get("current_file"):
            no_doc_text = "❗ Chưa chọn tài liệu. Vui lòng tải lên hoặc chọn một tài liệu trước." if is_vietnamese else "❗ No document selected. Please upload or select a document first."
            back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
            
            message = await self.send_safe_message(
                chat_id=update.effective_chat.id,
                text=no_doc_text,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(back_text, callback_data="menu_back")
                ]])
            )
            self.add_message_to_cleanup(user_id, message.message_id)
            
            return MAIN_MENU
        
        prompt = update.message.text
        file_name = self.user_data[user_id]["current_file"]
        file_ref = self.user_data[user_id]["files"][file_name]
        
        # Send typing action
        await update.message.chat.send_action(action="typing")
        
        # Indicate processing
        processing_text = f"⏳ Đang phân tích tài liệu: {file_name}...\n\nQuá trình này có thể mất một phút tùy thuộc vào kích thước và độ phức tạp của tài liệu." if is_vietnamese else f"⏳ Analyzing document: {file_name}...\n\nThis may take a minute depending on the size and complexity of your document."
        
        processing_message = await self.send_safe_message(
            chat_id=update.effective_chat.id,
            text=processing_text
        )
        self.add_message_to_cleanup(user_id, processing_message.message_id)
        
        # Analyze the document
        try:
            response = await self.analyze_document(file_ref, prompt)
            
            # Translate if needed
            if is_vietnamese:
                response = await self.translate_text(response, "vi")
            
            # Delete processing message
            try:
                await self.application.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=processing_message.message_id
                )
                # Remove from cleanup list
                if processing_message.message_id in self.user_data[user_id]["messages"]:
                    self.user_data[user_id]["messages"].remove(processing_message.message_id)
            except Exception as e:
                logger.info(f"Could not delete processing message: {e}")
            
            # Send results as a new message
            result_title = "📝 <b>Kết quả phân tích</b>" if is_vietnamese else "📝 <b>Analysis Results</b>"
            
            # Use the new split and send method
            await self.split_and_send_long_message(
                update=update,
                text=response,
                title=result_title
            )
            
            # Offer follow-up options
            if is_vietnamese:
                follow_up_text = "Bạn muốn làm gì tiếp theo?"
                ask_button = "🔍 Đặt câu hỏi khác"
                back_button = "🔙 Quay lại Menu"
            else:
                follow_up_text = "What would you like to do next?"
                ask_button = "🔍 Ask Another Question"
                back_button = "🔙 Back to Menu"
            
            keyboard = [
                [InlineKeyboardButton(ask_button, callback_data="menu_ask")],
                [InlineKeyboardButton(back_button, callback_data="menu_back")]
            ]
            
            message = await self.send_safe_message(
                chat_id=update.effective_chat.id,
                text=follow_up_text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            self.add_message_to_cleanup(user_id, message.message_id)
            
            return MAIN_MENU
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            
            # Delete processing message
            try:
                await self.application.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=processing_message.message_id
                )
                # Remove from cleanup list
                if processing_message.message_id in self.user_data[user_id]["messages"]:
                    self.user_data[user_id]["messages"].remove(processing_message.message_id)
            except Exception:
                pass
            
            error_text = f"❌ Lỗi khi phân tích tài liệu: {str(e)}\n\nVui lòng thử lại." if is_vietnamese else f"❌ Error analyzing document: {str(e)}\n\nPlease try again."
            back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
            
            message = await self.send_safe_message(
                chat_id=update.effective_chat.id,
                text=error_text,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(back_text, callback_data="menu_back")
                ]])
            )
            self.add_message_to_cleanup(user_id, message.message_id)
            
            return MAIN_MENU
    
    async def handle_pdf(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle PDF document uploads from the main menu."""
        user_id = update.effective_user.id
        
        # Determine language
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        # Get file information
        file = update.message.document
        file_name = file.file_name
        
        # Download the file
        downloading_text = f"⏳ Đang tải xuống {file_name}...\n\nVui lòng đợi trong khi tôi xử lý tài liệu của bạn." if is_vietnamese else f"⏳ Downloading {file_name}...\n\nPlease wait while I process your document."
        
        message = await self.send_safe_message(
            chat_id=update.effective_chat.id,
            text=downloading_text
        )
        self.add_message_to_cleanup(user_id, message.message_id)
        
        telegram_file = await context.bot.get_file(file.file_id)
        
        # Send typing action
        await update.message.chat.send_action(action="typing")
        
        try:
            # Download file content
            file_content = await self._download_telegram_file(telegram_file.file_path)
            
            # Upload to Google AI
            uploading_text = f"⏳ Đang tải {file_name} lên Google AI để phân tích..." if is_vietnamese else f"⏳ Uploading {file_name} to Google AI for analysis..."
            
            # Update the message instead of sending a new one
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=message.message_id,
                text=uploading_text
            )
            
            file_ref = await self.upload_pdf(file_content)
            
            # Store file reference
            self.user_data[user_id]["files"][file_name] = file_ref
            self.user_data[user_id]["current_file"] = file_name
            
            # Create quick analysis buttons
            if is_vietnamese:
                success_text = f"✅ Đã tải lên thành công: {file_name}\n\nĐây là tài liệu đã chọn của bạn. Bạn muốn làm gì?"
                summarize_button = "📝 Tóm tắt"
                key_points_button = "🔑 Điểm chính"
                ask_button = "❓ Đặt câu hỏi"
                back_button = "🔙 Quay lại Menu"
            else:
                success_text = f"✅ Successfully uploaded: {file_name}\n\nThis is now your selected document. What would you like to do?"
                summarize_button = "📝 Summarize"
                key_points_button = "🔑 Key Points"
                ask_button = "❓ Ask Question"
                back_button = "🔙 Back to Menu"
            
            keyboard = [
                [InlineKeyboardButton(summarize_button, callback_data="analyze_summarize")],
                [InlineKeyboardButton(key_points_button, callback_data="analyze_key_points")],
                [InlineKeyboardButton(ask_button, callback_data="menu_ask")],
                [InlineKeyboardButton(back_button, callback_data="menu_back")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Update the message instead of sending a new one
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=message.message_id,
                text=success_text,
                reply_markup=reply_markup
            )
            
            # Important: Return to MAIN_MENU state
            return MAIN_MENU
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            
            error_text = f"❌ Lỗi khi xử lý PDF: {str(e)}\n\nVui lòng thử lại." if is_vietnamese else f"❌ Error processing PDF: {str(e)}\n\nPlease try again."
            back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
            
            # Update the message instead of sending a new one
            await self.edit_safe_message(
                chat_id=update.effective_chat.id,
                message_id=message.message_id,
                text=error_text,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(back_text, callback_data="menu_back")
                ]])
            )
            
            return MAIN_MENU
    
    async def handle_pdf_in_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle PDF document uploads in upload state."""
        # Same as handle_pdf but returns to MAIN_MENU
        result = await self.handle_pdf(update, context)
        return MAIN_MENU
    
    async def handle_text_in_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle text messages in the main menu."""
        user_id = update.effective_user.id
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        text = "Vui lòng sử dụng các nút menu hoặc lệnh.\n\nGõ /help để xem các lệnh có sẵn." if is_vietnamese else "Please use the menu buttons or commands.\n\nType /help to see available commands."
        back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
        
        message = await self.send_safe_message(
            chat_id=update.effective_chat.id,
            text=text,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton(back_text, callback_data="menu_back")
            ]])
        )
        self.add_message_to_cleanup(user_id, message.message_id)
        
        return MAIN_MENU
    
    async def handle_text_in_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle text messages in upload state."""
        user_id = update.effective_user.id
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        text = "Tôi đang đợi bạn tải lên tài liệu PDF.\n\nVui lòng gửi cho tôi một tệp PDF hoặc gõ /back để quay lại menu chính." if is_vietnamese else "I'm waiting for you to upload a PDF document.\n\nPlease send me a PDF file or type /back to return to the main menu."
        back_text = "🔙 Quay lại Menu" if is_vietnamese else "🔙 Back to Menu"
        
        message = await self.send_safe_message(
            chat_id=update.effective_chat.id,
            text=text,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton(back_text, callback_data="menu_back")
            ]])
        )
        self.add_message_to_cleanup(user_id, message.message_id)
        
        return UPLOADING
    
    async def back_to_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return to the main menu."""
        return await self.show_menu(update, context)
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancel the conversation."""
        user_id = update.effective_user.id
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        text = "Đã hủy thao tác. Gõ /start để bắt đầu lại." if is_vietnamese else "Operation cancelled. Type /start to begin again."
        
        message = await self.send_safe_message(
            chat_id=update.effective_chat.id,
            text=text
        )
        self.add_message_to_cleanup(user_id, message.message_id)
        
        return ConversationHandler.END
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        user_id = update.effective_user.id
        is_vietnamese = self.user_data[user_id].get("language", "en") == "vi"
        
        if is_vietnamese:
            help_text = (
                "📚 <b>Trợ giúp Bot Phân tích PDF</b>\n\n"
                "<b>Lệnh cơ bản:</b>\n"
                "/start - Khởi động bot và hiển thị menu chính\n"
                "/menu - Hiển thị menu chính\n"
                "/help - Hiển thị tin nhắn trợ giúp này\n"
                "/language - Thay đổi ngôn ngữ\n"
                "/cancel - Hủy thao tác hiện tại\n\n"
                
                "<b>Cách sử dụng bot này:</b>\n"
                "1. Tải lên tài liệu PDF\n"
                "2. Chọn tùy chọn phân tích từ menu\n"
                "3. Đặt câu hỏi về tài liệu của bạn\n\n"
                
                "<b>Mẹo:</b>\n"
                "• Bạn có thể tải lên nhiều tài liệu và so sánh chúng\n"
                "• Để có kết quả tốt nhất, hãy sử dụng câu hỏi rõ ràng và cụ thể\n"
                "• Tài liệu lớn có thể mất nhiều thời gian hơn để phân tích\n"
            )
        else:
            help_text = (
                "📚 <b>PDF Analysis Bot Help</b>\n\n"
                "<b>Basic Commands:</b>\n"
                "/start - Start the bot and show main menu\n"
                "/menu - Show the main menu\n"
                "/help - Show this help message\n"
                "/language - Change language\n"
                "/cancel - Cancel current operation\n\n"
                
                "<b>How to use this bot:</b>\n"
                "1. Upload a PDF document\n"
                "2. Select analysis options from the menu\n"
                "3. Ask questions about your document\n\n"
                
                "<b>Tips:</b>\n"
                "• You can upload multiple documents and compare them\n"
                "• For best results, use clear and specific questions\n"
                "• Large documents may take longer to analyze\n"
            )
        
        message = await self.send_safe_message(
            chat_id=update.effective_chat.id,
            text=help_text
        )
        self.add_message_to_cleanup(user_id, message.message_id)
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the dispatcher."""
        logger.error(f"Exception while handling an update: {context.error}")
        
        if update and isinstance(update, Update) and update.effective_message:
            user_id = update.effective_user.id
            is_vietnamese = self.user_data.get(user_id, {}).get("language", "en") == "vi"
            
            error_text = "❌ Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn.\n\nVui lòng thử lại hoặc gõ /start để khởi động lại bot." if is_vietnamese else "❌ Sorry, an error occurred while processing your request.\n\nPlease try again or type /start to restart the bot."
            restart_text = "🔄 Khởi động lại" if is_vietnamese else "🔄 Restart"
            
            try:
                message = await self.send_safe_message(
                    chat_id=update.effective_chat.id,
                    text=error_text,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton(restart_text, callback_data="menu_back")
                    ]])
                )
                
                if user_id in self.user_data:
                    self.add_message_to_cleanup(user_id, message.message_id)
            except Exception as e:
                logger.error(f"Error sending error message: {e}")
    
    async def _download_telegram_file(self, file_path: str) -> bytes:
        """Download a file from Telegram."""
        async with httpx.AsyncClient() as client:
            response = await client.get(file_path)
            return response.content
    
    async def upload_pdf(self, pdf_content: bytes) -> Any:
        """Upload a PDF to Google AI."""
        # Create a BytesIO object from the PDF content
        pdf_io = io.BytesIO(pdf_content)
        
        # Upload the PDF using the File API
        uploaded_file = self.client.files.upload(
            file=pdf_io,
            config=dict(mime_type='application/pdf')
        )
        
        return uploaded_file
    
    async def analyze_document(self, file_ref: Any, prompt: str, model: str = "gemini-1.5-flash") -> str:
        """Analyze a document with a specific prompt."""
        # Generate content using the file and prompt
        response = self.client.models.generate_content(
            model=model,
            contents=[file_ref, prompt]
        )
        
        return response.text
    
    async def compare_documents(self, file_refs: List[Any], prompt: str, model: str = "gemini-1.5-flash") -> str:
        """Compare multiple documents with a specific prompt."""
        # Generate content using the files and prompt
        contents = file_refs + [prompt]
        response = self.client.models.generate_content(
            model=model,
            contents=contents
        )
        
        return response.text
    
    def _safe_callback_data(self, file_name: str) -> str:
        """Create a safe callback data string from a file name."""
        # Telegram has a 64 byte limit for callback data
        # We'll use a simple hash to ensure uniqueness
        if len(file_name) <= 32:
            return file_name
        
        # Create a simple hash for longer filenames
        hash_value = str(hash(file_name) % 10000)
        # Use first 28 chars + hash to stay under limit
        return file_name[:28] + "_" + hash_value
    
    def _get_filename_from_id(self, user_id: int, file_id: str) -> Optional[str]:
        """Get the original filename from a callback data ID."""
        # If the file_id is a complete filename, return it directly
        if file_id in self.user_data[user_id]["files"]:
            return file_id
        
        # Otherwise, try to match the shortened ID with the original filename
        for filename in self.user_data[user_id]["files"].keys():
            if file_id == self._safe_callback_data(filename):
                return filename
            
            # Handle the case where file_id is a prefix of the safe callback data
            safe_id = self._safe_callback_data(filename)
            if len(file_id) <= len(safe_id) and safe_id.startswith(file_id):
                return filename
        
        return None
    
    def run(self):
        """Run the bot."""
        print("Starting PDF Analysis Bot...")
        self.application.run_polling()

# Run the bot
if __name__ == "__main__":
    bot = SimplePDFBot()
    bot.run()