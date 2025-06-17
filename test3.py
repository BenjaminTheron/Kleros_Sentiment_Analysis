import certifi
import ssl
ssl._create_default_https_context = ssl._create_default_https_context.wrap_socket
ssl._create_default_https_context().load_default_certs()