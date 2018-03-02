from abc import ABCMeta, abstractmethod
import pandas as pd

def arg_check(x:int,X:list,Z:int,y=4,**z):
    #call by reference
    #(x:int,y=5:int,z) is ng
    print('x,y,z='+str(x)+','+str(y)+','+str(z))
    print(X)
    print(Z)
    x=[0]*5
    x[1]=1
    print(x)
    X.append(1)

class MetaClass(metaclass=ABCMeta):
    def __init__(self,arg:str):
        print(arg+' in constructor')
    @abstractmethod
    def get_name(self):
        raise NotImplementedError
    def print_const(self):
        print('in Metaclass.const')
    def print_field(self):
        print(self.name+' in Metaclass.field')

class Object(MetaClass):
    def __init__(self):
        super().__init__('unko')
        self.name='Object'
        self.print_const()
        self.print_field()
        self.get_name()
    def get_name(self):
        print('in Object.get_name')


class Creature(object):
    def __init__(self, level=1, weapon=None):
        self.level = level
        self.hp = 0
        self.mp = 0
        self.attack = 0
        self.defence = 0
        self.weapon = weapon
        self.job = "neet"

    def status(self):
        return "Job:{} | HP:{} | MP:{} | Atk:{} | Def:{} | Weapon:{}".format \
                (self.job, self.hp, self.mp, self.attack, self.defence, self.weapon)


class Warrior(Creature):
    def __init__(self, level):
        super().__init__(level)
        self.attack += 3 * level
        if self.weapon is None:
            self.weapon = "sword"
        if self.job == "neet":
            self.job = "Warrior"
        else: self.job += "Warrior"


class Magician(Creature):
    def __init__(self, level):
        super().__init__(level)
        self.mp += 4 * level
        if self.weapon is None:
            self.weapon = "rod"
        if self.job == "neet":
            self.job = "Magic"
        else: self.job += "Magic"

class MagicianNoSuper(Creature):
    def __init__(self, level):
        Creature.__init__(self,level)
        self.mp += 4 * level
        if self.weapon is None:
            self.weapon = "rod"
        if self.job == "neet":
            self.job = "Magic"
        else: self.job += "Magic"

class WarriorNoSuper(Creature):
    def __init__(self, level):
        Creature.__init__(self,level)
        self.attack += 4 * level
        if self.weapon is None:
            self.weapon = "sword"
        if self.job == "neet":
            self.job = "Warrior"
        else: self.job += "Warrior"


class WarriorMagic(Magician, Warrior):
    def __init__(self, level):
        # super().__init__(level)
        Warrior.__init__(self, level)
        Magician.__init__(self, level)

class WarriorMagicNoSuper(MagicianNoSuper, WarriorNoSuper):
    def __init__(self, level):
        # super().__init__(level)
        WarriorNoSuper.__init__(self, level)
        MagicianNoSuper.__init__(self, level)

class Super():
    def __init__(self,name):
        self.name='super'
        self.super_field='super_field'
    def print_name(self):
        print(self.name+' in Super.print_name()')
    def override1(self):
        print(self.name+' in Super.override1()')
        self.override2()
    def override2(self):
        print(self.name+' in Super.override2()')
class Sub(Super):
    def __init__(self,name):
        #Super.__init__(self,'')
        super().__init__('')
        self.name='sub'
        super().override1()
    def print_name(self):
        print(self.name+' in Sub.print_name()')
    def sub_method(self):
        print(self.name+' in Sub.sub_method()')
    def override1(self):
        print(self.name+' in Sub.override1()')
    def override2(self):
        print(self.name+' in Sub.override2()')

ob=Sub('')
ob.print_name()
ob.sub_method()

print(WarriorMagic(5).status())
print(WarriorMagic.mro())
print(WarriorMagicNoSuper(5).status())
print(WarriorMagicNoSuper.mro())
Object()
with open('../data_input.txt','wt') as fout:
    fout.write('header1,header2,list,header3\n')
    fout.write('1,2,'+'"'+str([1,3,3])+'"'+',3\n')
    fout.write('1,2,'+'"'+str([1,3,4])+'"'+',3\n')

X=[]
arg_check(1,X=X,Z=4)
arg_check(2,Z=5,X=X,O=3)
arg_check(L=4,Z=7,x=3,X=X)
#if arg_name ,order not needed




"""
Super.__init__ and self().__init__():
    sub in Sub.print_name()
    sub in Super.super_method()
    sub in Sub.sub_method()
Result2
Job:WarriorMagic | HP:0 | MP:20 | Atk:15 | Def:0 | Weapon:sword
[<class '__main__.WarriorMagic'>, <class '__main__.Magician'>, <class '__main__.Warrior'>, <class '__main__.Creature'>, <class 'object'>]
Job:Magic | HP:0 | MP:20 | Atk:0 | Def:0 | Weapon:rod
[<class '__main__.WarriorMagicNoSuper'>, <class '__main__.MagicianNoSuper'>, <class '__main__.WarriorNoSuper'>, <class '__main__.Creature'>, <class 'object'>]
"""
