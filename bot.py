import os
import telebot
import torch
import soundfile as sf
import numpy as np # إضافة numpy للتحقق من القيم
from zipvoice.luxvoice import LuxTTS

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
bot = telebot.TeleBot(TOKEN)

print("Loading LuxTTS Model...")
# تقليل عدد الـ threads إلى 2 لضمان الاستقرار على سيرفرات GitHub المجانية
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=2)

user_data = {}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "أهلاً بك! 🎙️\nأرسل بصمة صوتية (5 ثوانٍ)، ثم أرسل النص.")

@bot.message_handler(content_types=['voice', 'audio'])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id if message.voice else message.audio.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    prompt_path = f"prompt_{message.chat.id}.wav" # تغيير الصيغة لـ wav لضمان التوافق
    
    # حفظ الملف مؤقتاً
    with open("temp.ogg", 'wb') as f:
        f.write(downloaded_file)
    
    # استخدام ffmpeg لتحويل الملف لضمان جودته وتوافقه مع LuxTTS
    os.system(f"ffmpeg -y -i temp.ogg -ar 48000 {prompt_path}")
    
    user_data[message.chat.id] = {'prompt': prompt_path}
    bot.reply_to(message, "✅ تم حفظ البصمة بنجاح! أرسل النص الآن.")

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    chat_id = message.chat.id
    if chat_id in user_data:
        text = message.text
        if len(text) > 200: # حماية من النصوص الطويلة التي تسبب انهيار الذاكرة
             bot.reply_to(message, "⚠️ النص طويل جداً، حاول تقليله.")
             return

        bot.reply_to(message, "جاري المعالجة... ⏳")
        
        try:
            # استخدام محاولات استثنائية للـ CPU
            with torch.no_grad():
                encoded_prompt = lux_tts.encode_prompt(user_data[chat_id]['prompt'], duration=5, rms=0.01)
                final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)
                
                # تحويل آمن للمصفوفة
                final_wav_np = final_wav.detach().cpu().numpy().squeeze()
                
                # منع خطأ Floating Point عبر التأكد من القيم
                if np.isnan(final_wav_np).any() or np.isinf(final_wav_np).any():
                    raise ValueError("تعذر توليد الصوت بشكل صحيح (أرقام غير صالحة)")

                output_path = f"out_{chat_id}.wav"
                sf.write(output_path, final_wav_np, 48000)
            
            with open(output_path, 'rb') as audio:
                bot.send_voice(chat_id, audio)
            
            # تنظيف الملفات لعدم استهلاك مساحة السيرفر
            if os.path.exists(output_path): os.remove(output_path)
            
        except Exception as e:
            bot.reply_to(message, f"❌ خطأ تقني: {str(e)}")
    else:
        bot.reply_to(message, "⚠️ أرسل بصمة صوتك أولاً.")

bot.polling()
