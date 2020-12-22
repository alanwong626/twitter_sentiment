import os
os.environ['dev'] = '1'
from dashapp import dev_server, server

if __name__ == '__main__':
    #dev_server(debug=True)
    dev_server(debug=False)
    #server()
