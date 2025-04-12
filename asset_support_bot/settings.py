import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-default-key-for-development')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '*,localhost,127.0.0.1').split(',')

print("DB_NAME:", os.environ.get('DB_NAME'))
print("DB_USER:", os.environ.get('DB_USER'))
print("DB_HOST:", os.environ.get('DB_HOST'))

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party apps
    'rest_framework',
    'corsheaders',
    
    # Project apps
    'documents',
    'chatbot',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # CORS middleware
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = [
    "http://ml.presageinsights.ai",
    "https://ml.presageinsights.ai",
    "http://13.201.85.22",
    "https://13.201.85.22",
    "http://127.0.0.1",
]

CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

ROOT_URLCONF = 'asset_support_bot.urls'

# Inform Django it's behind a proxy
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Disable Django's SSL redirect (handled by NGINX)
SECURE_SSL_REDIRECT = False
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'asset_support_bot.wsgi.application'

# Database configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'asset_support_db'),
        'USER': os.environ.get('postgres', 'postgres'),
        'PASSWORD': os.environ.get('DB_PASSWORD', 'Sunilgiri@1#'),
        'HOST': os.environ.get('DB_HOST', 'asset-support-bot-db-1'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Media files
MEDIA_URL = 'media/'
# MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_ROOT = '/app/media'
DEFAULT_FILE_STORAGE = 'django.core.files.storage.FileSystemStorage'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS settings
CORS_ALLOW_ALL_ORIGINS = DEBUG  

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}

# Celery Configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
REDIS_HOST = os.environ.get('REDIS_HOST', 'asset-support-bot-redis-1')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
# Document upload settings
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024  # 20 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024  # 20 MB

# LLM Settings
LLM_MODEL_ID = os.getenv('LLM_MODEL_ID', 'mistralai/Mistral-7B-Instruct-v0.2')
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'all-MiniLM-L6-v2')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN', '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')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY', 'VrqhfV38Mxr8T90JzfEZ0cjINtm6Th5o')
# GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAN7hPbw5T0y20R3AEjyaiQAZL67oV5thA')

# Pinecone settings
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'asset-support-index')

# Document processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'chatbot': {  # Match this with the logger name in the code
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}