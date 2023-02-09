
from client import Client
from Net.Server import Server
import os
from CVSOperator import CSVOperator
import threading 

model_path = "\\diabetes\\"
data_set_o = "dataset\\diabetes\\pre_operate\\step2\\"
data_sets = os.listdir(data_set_o)

clients = []
class MyThread(threading.Thread):  # 重写threading.Thread类，加入获取返回值的函数

    def __init__(self, c):
        threading.Thread.__init__(self)
        self.client = c            # 初始化传入的client
        self.result = False

    def run(self):                    
        self.result = self.client.training_process()  
                                    

    def get_result(self):  
        return self.result

def multi_thread():
    print("start")
    threads = []           # 定义一个线程组
    for thread in threads: # 每个线程组start
        thread.start()

    for thread in threads: # 每个线程组join
        thread.join()

    list = []
    for thread in threads:
        list.append(thread.get_result())  # 每个线程返回结果(result)加入列表中

    print("end")
    return list  # 返回多线程返回的结果组成的列表


echo = 4000  # 迭代的总次数

total_clients = len(data_sets)

center_server = Server()
for i in range(total_clients):
    clients.append(Client(data_set_o+data_sets[i], 0, i, data_sets[i]))

center_server.clients_num = total_clients
center_server.clients_pool = clients
time_cost_server = 0

for e in range(echo):
    t = []
    for c in clients:    
        t.append(MyThread(c)) # 加载多线程
    sum_loss = 0
    cn = 0
    finished = False
    for c in t:
        c.start() # 开始线程
    while finished is False:
        for c in t:
            if c.get_result():
                c.join()
                center_server.grad_pool.append("a")
                if len(center_server.grad_pool) == center_server.clients_num:
                    finished = True
            
                
                
    print("---------")
    print(e)
    print("----------------")
    center_server.aggregate_weight(clients)
    time_cost_server = center_server.time_cost/(e+1)
    time_cost_client = 0
    for c in clients:
        #if e >= 1:
            c.test(e+1)
            time_cost_client += c.time_cost
csv_writer = CSVOperator("result/time_analysis/timecost_client.csv",'w')
csv_writer.write_row([[time_cost_client]])
csv_writer.end()
csv_writer = CSVOperator("result/time_analysis/timecost_server.csv",'w')
csv_writer.write_row([[time_cost_server]])
csv_writer.end()