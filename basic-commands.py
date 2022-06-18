def default():
    print("Not a valid choice...")


def basic():
    print("\n ~ This calculator perform basic operations between two numbers")
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
    print("Here are the numbers\nadded: %s \nsubtracted: %s \nmultiplied: %s \ndivided: %s" %(str(round(num1+num2,2)), str(round(num1-num2,2)), str(round(num1*num2,2)), str(round(num1/num2,3))))
def compound():
    print("\n ~ This is a compound interest calculator")
    startValue = round(float(input("Enter the start value (£): ")), 2)
    interest = round(float(input("Enter the interest rate (%): ")), 2)
    years = round(float(input("Enter the years of interest: ")), 2)
    endValue = round((startValue * (1+interest/100)**years), 2)
    print("%s will be worth %s (after %s years at %s%% interest)" %(str(startValue), str(endValue), str(years), str(interest) ) )
def income():
    taxBrackets = [12570, 50270, 150000]
    taxBracketRate = [0.20, 0.20, 0.05]
    netIncome = 0
    tax = 0
    print("\n ~ This is an Income tax calculator for the UK")
    grossIncome = float(input("What is your gross income (income before taxes or other deductions)?: "))
    for i in range (0, len(taxBrackets)):
        if grossIncome > taxBrackets[i]:
            tax = tax + (grossIncome - taxBrackets[i]) * taxBracketRate[i]
            print(tax)
    netIncome = grossIncome - tax
    print("£%s gross income after Income tax is a net income of £%s (paying tax of £%s)" % (str(round(grossIncome, 2)), str(round(netIncome, 2)), str(round(tax, 2))) )

calculators = {
    "1": basic,
    "2": compound,
    "3": income
}

calculate = "y"
while calculate.lower() == "y" or calculate.lower() == "yes":
    choice = input("What calculator do you want?\n1 - Basic Arithmetic\n2 - Compound Interest\n3 - Income Tax\n - ")
    calculators.get(choice, default)()
    calculate = input("\nIf you want to perform another calculation please enter ""yes"" or ""y"": ")