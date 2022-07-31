# import test
from http import server
import client
import Server
import noise_generator
import neural_net_work
import _thread
import data_generator
import threading



clients = []
K = 5
echo = 10
total_clients = 8
center_server = Server.Server()
for i in range(total_clients):
    clients.append(client.Client(i))
# center_server.clients_pool = clients
center_server.clients_num = total_clients
center_server.clients_pool = clients
data_generator.allocate_data(clients=clients)
for e in range(echo):
    sum_loss = 0
    for c in clients:
        center_server.grad_pool.append(c.learning_algorithm())
        sum_loss += c.sum_loss
    # print(sum_loss)
    
    print(e)  
    center_server.aggregate_weight(clients)
for c in clients:
    c.save_model()



    
