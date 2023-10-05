import logging as lg

def logger(file_name , massage):
    lg.basicConfig(filename= file_name,
               level=lg.INFO,format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",filemode='a+')
    console_log = lg.StreamHandler()
    console_log.setLevel(lg.INFO)
    format_1 =  lg.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
    console_log.setFormatter(format_1)
    lg.getLogger('').addHandler(console_log)
    logger1 = lg.getLogger("Nasir")
    logger1.info(massage)

'''
if __name__ == "__main__":
    logger('log\ model.txt','error occured')
'''