# Gunicorn configuration file
bind = "0.0.0.0:5000"
workers = 2
timeout = 300  # 5 minutes for model loading and processing
keepalive = 2
max_requests = 1000
max_requests_jitter = 50
preload_app = True
worker_class = "sync"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
