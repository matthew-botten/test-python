def default():
    print("Not a valid choice...")

def basic():
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
    print("Here are the numbers\nadded: %s \nsubtracted: %s \nmultiplied: %s \ndivided: %s" %(str(round(num1+num2,2)), str(round(num1-num2,2)), str(round(num1*num2,2)), str(round(num1/num2,3))))
def compound():
    print("This is a compound interest calculator\n")
    startValue = round(float(input("Enter the start value (£): ")), 2)
    interest = round(float(input("Enter the interest rate (%): ")), 2)
    years = round(float(input("Enter the years of interest: ")), 2)
    endValue = round((startValue * (1+interest/100)**years), 2)
    print("%s will be worth %s (after %s years at %s%% interest)" %(str(startValue), str(endValue), str(years), str(interest) ) )
def income():
    print("The Income tax calculator is not yet functional, sorry!")


choice = input("What calculator do you want?\n1 - Basic Arithmetic\n2 - Compound Interest\n3 - Income Tax\n - ")
calculators = {
    "1": basic,
    "2": compound,
    "3": income
}

calculators.get(choice, default)()