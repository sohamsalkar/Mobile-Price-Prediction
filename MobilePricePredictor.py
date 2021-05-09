import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import matplotlib.pyplot as plt
import seaborn as sns

from tkinter import *
root = Tk()

def ins():
	data = pd.read_csv("data.csv")
	y = data['price_range']
	x = data.drop('price_range',axis=1)
	x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.2,random_state=101,stratify=y)
	
	dt = DecisionTreeClassifier(random_state=101)
	dt_model = dt.fit(x_train, y_train)

	lr = LogisticRegression(multi_class = 'multinomial', solver='sag', max_iter=10000)
	lr.fit(x_train, y_train)

	model_knn = KNeighborsClassifier(n_neighbors=3)
	model_knn.fit(x_train, y_train)
	
	
	x_test=[[0]*20]*1

	x_test[0][0] = float(u0.get())
	x_test[0][1] = float(u1.get())
	x_test[0][2] = float(u2.get())
	x_test[0][3] = float(u3.get())
	x_test[0][4] = float(u4.get())
	x_test[0][5] = float(u5.get())
	x_test[0][6] = float(u6.get())
	x_test[0][7] = float(u7.get())
	x_test[0][8] = float(u8.get())
	x_test[0][9] = float(u9.get())
	x_test[0][10] = float(v0.get())
	x_test[0][11] = float(v1.get())
	x_test[0][12] = float(v2.get())
	x_test[0][13] = float(v3.get())
	x_test[0][14] = float(v4.get())
	x_test[0][15] = float(v5.get())
	x_test[0][16] = float(v6.get())
	x_test[0][17] = float(v7.get())
	x_test[0][18] = float(v8.get())
	x_test[0][19] = float(v9.get())

	y_pred_dt = dt.predict(x_test)
	blank1.insert(0,*y_pred_dt)
	print(*y_pred_dt)
	y_pred_knn = model_knn.predict(x_test)  
	blank2.insert(0,*y_pred_knn)
	print(*y_pred_knn)
	y_pred_knn = model_knn.predict(x_test)  
	blank3.insert(0,*y_pred_knn)
	print(*y_pred_knn)

root.geometry("1450x800")
root.maxsize(2000,1000)
root.minsize(900,500)
root.title("Mobile Price Prediction - Group 10")

head = Label(root, text="range 0: ₹4000-₹10000\n\nrange 1: ₹10000-₹16000\n\nrange 2: ₹16000-₹22000\n\nrange 3: ₹22000+", font="comicsansms 13 bold",bg='#e06d1f', padx="10",pady="10")
head.pack(fill='x')

f1 = Frame(root,bg='#ffa938',borderwidth=5)
f1.pack(side=LEFT)

root2 = Frame(root,bg='#ffa938',borderwidth=5)
root2.pack(side=LEFT)

root1 = Frame(root,bg='#ffa938',borderwidth=5)
root1.pack(side=RIGHT)

f2 = Frame(root,bg='#ffa938',borderwidth=5)
f2.pack(side=RIGHT)

f3 = Frame(root,bg='#d18b11',borderwidth=5)
f3.pack(side=BOTTOM)

l0 = Label(f1,text='Battery Power',bg='#ffa938',padx=25,pady=20)
l0.pack()
u0 = StringVar()
u0_entry = Entry(root2,textvariable=u0)
u0_entry.pack(anchor='nw',padx=24,pady=17)

l1 = Label(f1,text='Bluetooth ?',bg='#ffa938',padx=25,pady=20)
l1.pack()
u1 = StringVar()
u1_entry = Entry(root2,textvariable=u1)
u1_entry.pack(anchor='nw',padx=24,pady=17)

l2 = Label(f1,text='Clock Speed MHz',bg='#ffa938',padx=25,pady=20)
l2.pack()
u2 = StringVar()
u2_entry = Entry(root2,textvariable=u2)
u2_entry.pack(anchor='nw',padx=24,pady=17)


l3 = Label(f1,text='Dual Sim ?',bg='#ffa938',padx=25,pady=20)
l3.pack()
u3 = StringVar()
u3_entry = Entry(root2,textvariable=u3)
u3_entry.pack(anchor='nw',padx=24,pady=17)

l4 = Label(f1,text='Front Camera',bg='#ffa938',padx=25,pady=20)
l4.pack()
u4 = StringVar()
u4_entry = Entry(root2,textvariable=u4)
u4_entry.pack(anchor='nw',padx=24,pady=17)

l5 = Label(f1,text='4G enabled?',bg='#ffa938',padx=25,pady=20)
l5.pack()
u5 = StringVar()
u5_entry = Entry(root2,textvariable=u5)
u5_entry.pack(anchor='nw',padx=24,pady=17)

l6 = Label(f1,text='Internal Memory (Gb)',bg='#ffa938',padx=25,pady=20)
l6.pack()
u6 = StringVar()
u6_entry = Entry(root2,textvariable=u6)
u6_entry.pack(anchor='nw',padx=24,pady=17)

l7 = Label(f1,text='Depth (cm)',bg='#ffa938',padx=25,pady=20)
l7.pack()
u7 = StringVar()
u7_entry = Entry(root2,textvariable=u7)
u7_entry.pack(anchor='nw',padx=24,pady=17)

l8 = Label(f1,text='Mobile Weight (gm)',bg='#ffa938',padx=25,pady=20)
l8.pack()
u8 = StringVar()
u8_entry = Entry(root2,textvariable=u8)
u8_entry.pack(anchor='nw',padx=24,pady=17)

l9 = Label(f1,text='no. of Cores',bg='#ffa938',padx=25,pady=20)
l9.pack()
u9 = StringVar()
u9_entry = Entry(root2,textvariable=u9)
u9_entry.pack(anchor='nw',padx=24,pady=17)


v0 = StringVar()
v0_entry = Entry(root1,textvariable=v0)
v0_entry.pack(anchor='ne',padx=24,pady=17)
t0 = Label(f2,text='Primary Camera (MP)',bg='#ffa938',padx=25,pady=20)
t0.pack()

v1 = StringVar()
v1_entry = Entry(root1,textvariable=v1)
v1_entry.pack(anchor='ne',padx=24,pady=17)
t1 = Label(f2,text='Pixel Height ',bg='#ffa938',padx=25,pady=20)
t1.pack()

v2 = StringVar()
v2_entry = Entry(root1,textvariable=v2)
v2_entry.pack(anchor='ne',padx=24,pady=17)
t2 = Label(f2,text='Pixel Width ',bg='#ffa938',padx=25,pady=20)
t2.pack()

v3 = StringVar()
v3_entry = Entry(root1,textvariable=v3)
v3_entry.pack(anchor='ne',padx=24,pady=17)
t3 = Label(f2,text='Ram (Mb)',bg='#ffa938',padx=25,pady=20)
t3.pack()

v4 = StringVar()
v4_entry = Entry(root1,textvariable=v4)
v4_entry.pack(anchor='ne',padx=24,pady=17)
t4 = Label(f2,text='Screen Height (cm)',bg='#ffa938',padx=25,pady=20)
t4.pack()

v5 = StringVar()
v5_entry = Entry(root1,textvariable=v5)
v5_entry.pack(anchor='ne',padx=24,pady=17)
t5 = Label(f2,text='Screen Width (cm)',bg='#ffa938',padx=25,pady=20)
t5.pack()

v6 = StringVar()
v6_entry = Entry(root1,textvariable=v6)
v6_entry.pack(anchor='ne',padx=24,pady=17)
t6 = Label(f2,text='TalkTime (hr)',bg='#ffa938',padx=25,pady=20)
t6.pack()

v7 = StringVar()
v7_entry = Entry(root1,textvariable=v7)
v7_entry.pack(anchor='ne',padx=24,pady=17)
t7 = Label(f2,text='3G enabled?',bg='#ffa938',padx=25,pady=20)
t7.pack()

v8 = StringVar()
v8_entry = Entry(root1,textvariable=v8)
v8_entry.pack(anchor='ne',padx=24,pady=17)
t8 = Label(f2,text='Touch Screen ?',bg='#ffa938',padx=25,pady=20)
t8.pack()

v9 = StringVar()
v9_entry = Entry(root1,textvariable=v9)
v9_entry.pack(anchor='ne',padx=24,pady=17)
t9 = Label(f2,text='Wi-Fi ?',bg='#ffa938',padx=25,pady=20)
t9.pack()

p = Label(f3,text='Click Submit to find range',font="comicsansms 10 bold",pady=20)
p.pack()
b1 = Button(f3,fg='black',bg='#d18b11',width=27,text="Submit",command=ins)
b1.pack()

p1 = Label(f3,text='The Predicted Mobile price range\nby Decision Tree algorithm',font="comicsansms 20 bold",fg='black',bg='#d18b11')
p1.pack()
blank1 = Entry(f3)
blank1.pack(pady=20)

p1 = Label(f3,text='The Predicted Mobile price range\nby K-Nearest-Neighbour algorithm',font="comicsansms 20 bold",fg='black',bg='#d18b11')
p1.pack()
blank2 = Entry(f3)
blank2.pack(pady=20)

p1 = Label(f3,text='The Predicted Mobile price range\nby Logistic regression algorithm\n[max accuracy]',font="comicsansms 20 bold",fg='black',bg='#d18b11')
p1.pack()
blank3 = Entry(f3)
blank3.pack(pady=20)

root.mainloop()