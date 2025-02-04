from  threading import Thread

from tg_asadscalperbot import main as tg_bot
from ..data_collecting.BybitScalper import main as reports

Thread(target = tg_bot).start() 
Thread(target = reports).start()