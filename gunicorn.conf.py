# Gunicorn configuration file
import os

# Render requires binding to 0.0.0.0:$PORT
port = os.environ.get('PORT', '10000')
bind = f"0.0.0.0:{port}"

workers = 1  # Reduced for memory efficiency
timeout = 300  # 5 minutes for model loading and processing
keepalive = 2
max_requests = 500  # Reduced for memory management
max_requests_jitter = 50
preload_app = True
worker_class = "sync"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
