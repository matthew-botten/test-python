def add(value,interest,years):
    rate = 1 + (interest/100)
    endValue = value * rate**years
    return round(endValue,2)
def default():
    print("Not a valid choice...")


def basic():
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
    print("Here are the numbers\nadded: %s \nsubtracted: %s \nmultiplied: %s \ndivided: %s" %(str(round(num1+num2,2)), str(round(num1-num2,2)), str(round(num1*num2,2)), str(round(num1/num2,2))))
def compound():
    print("This is a compound interest calculator\n")
    print("£" + str(add( int(input("Enter the original value (£): ")),int(input("Enter the interest rate (%): ")),int(input("Enter the years of interest: ")) ))  )
def income():
    print("The Income tax calculator is not yet functional, sorry!")

choice = input("What calculator do you want?\n1 - Basic Arithmetic\n2 - Compound Interest\n3 - Income Tax\n - ")
calculators = {
    "1": basic,
    "2": compound,
    "3": income
}

calculators.get(choice, default)()