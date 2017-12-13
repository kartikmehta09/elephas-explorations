# some tests for debugging and making sure everything is working correctly
from pyspark import SparkContext
import sys
import socket
from tensorflow.python.client import device_lib

sc = SparkContext()

import keras
print " this is the keras version: ", keras.__version__
print "----this code is running on "
print socket.gethostname()
print "----it is using python from "
print sys.executable
print "----here are the devices visible to me:"
print device_lib.list_local_devices()
print "----"

# construct this test RDD to test what workers are available
rdd = sc.parallelize(range(0, 1000), 48)
rdd_hostnames = rdd.map(lambda id: [socket.gethostname()])
hosts_used = rdd_hostnames.reduce(lambda x, y: x + y)
unique_hosts = set(hosts_used)

print "----here are the hosts I see in the cluster:"
print(unique_hosts)
print "----and here are their IP adresses:"
print [ socket.gethostbyname(h) for h in unique_hosts ]