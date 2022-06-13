def add(value,interest,years):
    rate = 1 + (interest/100)
    endValue = value * rate**years
    return endValue
print("This is a compound interest calculator\n")
print( add( int(input("Enter the original value (£): ")),int(input("Enter the interest rate (%): ")),int(input("Enter the years of interest: ")) )  )
