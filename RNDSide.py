from pylogix import PLC
import sqlite3
import time
import sys
from struct import pack, unpack_from
# Setup the PLC object with initial parameters
# Change to your plc ip address, and slot, default is 0, shown for clarity
comm = PLC('192.168.0.88', 0)


demoDB= sqlite3.connect("demo.db")
my_cur = demoDB.cursor()

#my_cur.execute("CREATE TABLE PressureTimeGraph(time,pressure)")
#res = my_cur.execute("SELECT name FROM sqlite_master")
#res.fetchone() is None
#my_cur.execute(""" INSERT INTO PressureTimeGraph""")




#def read_from_plc():
   # returnTags= comm.Read(tags)


# this is using list comphrension
# for every tag (r) in ret( from the tags list) , the return statement is returning an array of values of the tag(r.Value for the tag)
 #can be combined with conditional if needed
 #   Only accept items that are not "apple":
 # EX:  newlist = [x for x in fruits if x != "apple"]

 #   return [r.Value for r in ret]



tag_list= 'DemoTag'
# Read returns Response class (.TagName, .Value, .Status)
ret = comm.Read(tag_list)
print(ret)


# Close Open Connection to the PLC
comm.Close()


#wonder if i can open multiple requests at the same time,