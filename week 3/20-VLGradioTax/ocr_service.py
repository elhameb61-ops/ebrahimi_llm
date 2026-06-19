import base64
import httpx
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def get_openrouter_client():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key='',
        base_url="https://openrouter.ai/api/v1"
    )

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_mime_type(image_path: str) -> str:
    extension = image_path.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    return mime_types.get(extension, 'image/jpeg')

def extract_national_code_with_langchain(image_path: str) -> tuple:
    
    try:
        llm = get_openrouter_client()
        
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        messages = [
            SystemMessage(content="""شما یک سیستم OCR هستید که اطلاعات کارت ملی ایرانی را استخراج می‌کنید.
فقط کد ملی ۱۰ رقمی را استخراج کنید.
پاسخ را فقط به صورت یک عدد ۱۰ رقمی بدهید.
اگر کد ملی پیدا نشد، کلمه "NOT_FOUND" را برگردانید.
                          کد ملی را به صورت عدد و فقط به صورت انگلیسی را برگردان. یعنی اعداد
                          ۱۲۷۰۷۶۵۱۰۸ 
                          نباشد. یعنی به صورت
                          1270765108
                          باشد.
                          """),
            # HumanMessage(content=[
            #     {
            #         "type": "text",
            #         "text": "لطفاً کد ملی را از این تصویر کارت ملی استخراج کنید. فقط عدد ۱۰ رقمی کد ملی را برگردانید."
            #     },
            #     {
            #         "type": "image_url",
            #         "image_url": {
            #             "url": f"data:{mime_type};base64,{base64_image}"
            #         }
            #     }
            # ])
            # مثال ۱ (تصویر نامرتبط)
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "لطفاً کد ملی را از این تصویر استخراج کن."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,IMAGE_1"
                    }
                }
            ]),
            SystemMessage(content="کد ملی در تصویر یافت نشد"),

            # مثال ۲ (تصویر نامرتبط دیگر)
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "لطفاً کد ملی را از این تصویر استخراج کن."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,IMAGE_2"
                    }
                }
            ]),
            SystemMessage(content="کد ملی در تصویر یافت نشد"),

            # تصویر واقعی (ورودی اصلی)
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "لطفاً کد ملی را از این تصویر کارت ملی استخراج کن. فقط عدد ۱۰ رقمی را برگردان."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ])
        ]
        
        response = llm.invoke(messages)
        result = response.content.strip()
        
        numbers = re.findall(r'\d+', result)
        for num in numbers:
            if len(num) == 10:
                return num, None
        
        if "NOT_FOUND" in result:
            return None, "کد ملی در تصویر یافت نشد"
        
        return None, "کد ملی معتبر در تصویر یافت نشد"
        
    except Exception as e:
        return None, f"خطا در پردازش تصویر: {str(e)}"

def extract_national_code_with_httpx(image_path: str, api_key: str, model: str = "openai/gpt-4o") -> tuple:
    if not api_key:
        return None, "کلید API وارد نشده است"
    
    try:
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "Tax Calculator App",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": """شما یک سیستم OCR هستید که اطلاعات کارت ملی ایرانی را استخراج می‌کنید.
فقط کد ملی ۱۰ رقمی را استخراج کنید.
پاسخ را فقط به صورت یک عدد ۱۰ رقمی بدهید.
اگر کد ملی پیدا نشد، کلمه "NOT_FOUND" را برگردانید.
کد ملی را فقط به صورت انگلیسی استخراج کن. حتی اگر فارسی باشد
مثلا کد ملی نباید به این صورت باشد:
۰۹۸۰۸۰۹۸۱۰
بله باید به این صورت باشد:
0980809810
"""

                },
                # {
                #     "role": "user",
                #     "content": [
                #         {
                #             "type": "text",
                #             "text": "لطفاً کد ملی را از این تصویر کارت ملی استخراج کنید. فقط عدد ۱۰ رقمی کد ملی را برگردانید."
                #         },
                #         {
                #             "type": "image_url",
                #             "image_url": {
                #                 "url": f"data:{mime_type};base64,{base64_image}"
                #             }
                #         }
                #     ]
                # }
                #   مثال 1 (تصویر نامرتبط)
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "لطفاً کد ملی را از این تصویر استخراج کن."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,IMAGE_1"
                            }
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": "NOT_FOUND"
                },

                # 🎭 مثال 2 (تصویر نامرتبط دیگر)
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "لطفاً کد ملی را از این تصویر استخراج کن."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,IMAGE_2"
                            }
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": "NOT_FOUND"
                },

                #  ورودی واقعی
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "لطفاً کد ملی را از این تصویر کارت ملی استخراج کن. فقط عدد ۱۰ رقمی را برگردان."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 50
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
        result_json = response.json()
        result = result_json["choices"][0]["message"]["content"].strip()
        
        numbers = re.findall(r'\d+', result)
        for num in numbers:
            if len(num) == 10:
                return num, None
        
        if "NOT_FOUND" in result:
            return None, "کد ملی در تصویر یافت نشد"
        
        return None, "کد ملی معتبر در تصویر یافت نشد"
        
    except httpx.HTTPStatusError as e:
        return None, f"خطای HTTP: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return None, f"خطا در پردازش تصویر: {str(e)}"

def extract_national_code_from_image(image_path: str) -> tuple:
    return extract_national_code_with_langchain(image_path)

def validate_national_code(national_code: str) -> bool:
    if not national_code or len(national_code) != 10:
        return False
    
    if not national_code.isdigit():
        return False
    
    if national_code == national_code[0] * 10:
        return False
    
    try:
        check = int(national_code[9])
        total = sum(int(national_code[i]) * (10 - i) for i in range(9))
        remainder = total % 11
        
        if remainder < 2:
            return check == remainder
        else:
            return check == (11 - remainder)
    except:
        return False

AVAILABLE_MODELS = [
    ("openai/gpt-4o", "GPT-4o (OpenAI)"),
    ("openai/gpt-4o-mini", "GPT-4o Mini (OpenAI)"),
    ("openai/gpt-4-turbo", "GPT-4 Turbo (OpenAI)"),
    ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet (Anthropic)"),
    ("anthropic/claude-3-opus", "Claude 3 Opus (Anthropic)"),
    ("google/gemini-pro-vision", "Gemini Pro Vision (Google)"),
    ("google/gemini-pro-1.5", "Gemini 1.5 Pro (Google)"),
    ("meta-llama/llama-3.2-90b-vision-instruct", "Llama 3.2 90B Vision (Meta)"),
]

def get_model_choices():
    return [model[0] for model in AVAILABLE_MODELS]

def get_model_labels():
    return {model[0]: model[1] for model in AVAILABLE_MODELS}