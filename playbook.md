/* The info will definitely need to be organized */

### Templates:
```cpp
//sstream template for parsing input
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;

/* add func type */ solution(/* fill in parameters */)
{
    return /* add return value */; 
}

int main()
{
    int tests = 0;
    /* add input type */ input_values = /* init */;
    string input_string = "";
    vector</* add input type */> input_vector;

    /* add necessary variables */

    cin >> tests;
    for(int i = 0; i < tests; i++)
    {
        cin.ignore();
        getline(cin, input_string);
        istringstream iss(input_string);

        while(iss >> input_values)
            input_vector.push_back(input_values); 

        /* add necessary code */
        cout << solution(/* fill in parameters */) << endl; 
        input_vector.clear();
    }
    return 0;
}
```

##Algorithms:

From Past ACM Cheat Sheet 2014:
Bitwise Operators 

| operation | syntax |
| --- | --- |
| bitwise AND | `&` |
| bitwise OR | `|` |
| bitwise XOR | `^` |
| left shift | `<<` |
| right shift | `>>` |
| complement | `~` |
| all 1’s in binary | `-1` |

### Macro to check if a bit is set
```cpp
#define CHECK_BIT(variable, position) ((variable) & (1 << (position)))
```
and use it like this
```cpp
CHECK_BIT(temp, 3)
```
### Inversions
```cpp
// Count the collisions problem
// input:  RRRLLL
// output: 9
#include<iostream>
using namespace std;
int main() {
    int count_r,total;
    string input;
    while(getline(cin,input)) {
        count_r = total = 0;
        for(int i=0; i < input.length();++i)
            if(input[i] == 'R') count_r++;
            else total += count_r;
    cout << total << endl;
    }
    return 0;
}
```
Mathematics
Algebra 
Sum of Powers

### Fast Exponentiation
```cpp 
// This is a very good way to reduce the overhead of the <cmath> library pow function   
double pow(double a, int n) {
    if(n == 0) return 1;
    if(n == 1) return a;
    double t = pow(a, n/2);
    return t * t * pow(a, n%2);
}
```
### Greatest Common Divisor (GCD)           
```cpp
int gcd(int a, int b) {
    while(b){int r = a % b; a = b; b = r;} 
    return a;    
}
```

#### Euclidian Algorithm
```cpp
while (a > 0 && b > 0)
a > b ? a = a - (2*b); : b = b - (2*a);
a > b ? return a; : return b;
```
Primes
Sieve of Eratosthenes
```cpp
#include<vector>
#include<cmath>
#include<iostream>
using namespace std;

vector<bool> num;

void sieve(int n) {
  num[0]=0;
  num[1]=0;
  int m=(int)sqrt(n);
  for (int i=2; i<=m; i++) 
    if (num[i])
      for (int k=i*i; k<=n; k+=i) 
        num[k]=0;
}
int main() {
    num = vector<bool>(1000000,1);
    sieve(1000000);
    for(int i=0;i<num.size();++i)
        if(num[i]) cout<<i<<endl;
}
```

O(sqrt(x)) Exhaustive Primality Test

```cpp
// Use the Sieve for numbers 1-1000000, 
// but anything larger than 1000000, use IsPrimeSlow
#include <cmath>
#define EPS 1e-7
typedef long long LL;
bool IsPrimeSlow (LL x)
{
  if(x<=1) return false;
  if(x<=3) return true;
  if (!(x%2) || !(x%3)) return false;
  LL s=(LL)(sqrt((double)(x))+EPS);
  for(LL i=5;i<=s;i+=6)
  {
    if (!(x%i) || !(x%(i+2))) return false;
  }
  return true;
}
```

handling input
Eating newline characters before getline()
input:
3
some string with spaces
another string with spaces
third string with spaces

output:
some string with spaces
another string with spaces
third string with spaces

solution 1:
```cpp
#include<iostream>
//#include <sstream>
using namespace std;
int main() {
  int N;
  string s;
//getline(cin, s);
//stringstream ss (s); 
  cin >> N;
//ss >> N;
  cin.get(); // get the newline character that is after the number 3,
 // otherwise getline will get an empty string the first time
  while(N--) {
    getline(cin, s);
    cout << s << endl;
  }
  return 0;
}
```
solution 2:
```cpp
#include<iostream>
#include <sstream>
using namespace std;
int main() {
  int N;
  string s;
  getline(cin, s);
  stringstream ss (s); 
  ss >> N;
  while(N--) {
    getline(cin, s);
    cout << s << endl;
  }
  return 0;
}
```
Set stream to not ignore whitespace
input:
a b c d

output:
0 1 2 3 4 5 6 7

solution 1 (faster):
```cpp
#include<iostream>
using namespace std;
int main(){
  char c;
  for(int i=0; cin>>noskipws>>c ;++i)
    cout << i << “ “;
  return 0;
}
```
solution 2:
```cpp
#include<iostream>
using namespace std;
int main(){
   string s;
   getline(cin,s);  
   for(int i=0; i < string.length() ;++i)
     cout << i << “ “;
   return 0;
}
```
toBase
```cpp
// Assumes “int number” parameter is base10
string toBase(int number, int base) {
  string vals = "0123456789ABCDEFGHIJLMNOP";
  string result = "";
  while(number) {
    result = vals[number%base] + result;
    number /= base;
  }
  return result;
}
isPalindrome
bool isPalindrome(string s) {
  for(int i=0,max=s.length()/2,len=s.length()-1;i<max;++i)
    if(s[i]!=s[len-i]) return false;
  return true;
}
```
Finding all the Armstrong Numbers up to 1000
/*
Problem: Find all Armstrong numbers up to 1000

An Armstrong number is a number that is the sum of its
own digits, each raise to the power of the number of
digits.

Steps:

1. Begin with an n-digit number
2. Raise each digit to the nth power and compute the sum
3. If the sum is same as the n-digit number, it is an Armstrong number (i.e. for 3-digit number 153: 1^3 + 5^3 + 3^3 = 153)
4. Continue loop and test the next number
*/
```cpp
#include <iostream>
using namespace std;

bool isArmstrong(int x);
int pow(int n, int power);
int digits(long int n);

int main() 
{
    int min = 0, max = 1000;
    cout << "Program to find Armstrong numbers..." 
         << endl;
    cout << "Enter a range for the Armstrong number..." 
         << endl;
    cout << "Enter lowest value to test: ";
    cin >> min;
    cout << "Enter highest value to test: ";
    cin >> max;
    cout << endl;

    for (int i = min; i <= max; ++i)
        if (isArmstrong(i)) 
            cout << i << " ";

    cout << endl;
    return 0;
}

bool isArmstrong(int x)
{
    int n = x;
    int d = digits(x);
    int y = 0, z = x;

    while (z > 0)
    {
        x = z % 10;
        y = y + pow(x, d);
        z /= 10;
    }

    if (y == n) return true;
    else return false;
}

int pow(int n, int power)
{
    if (power == 1) return n;
    else return n * pow(n, power - 1);
}

int digits(long int n)
{
    if (n < 10) return 1;
    else return 1 + digits(n / 10);
}
```
```
/*
OUTPUT

Program to find Armstrong numbers...
Enter a range for the Armstrong number...
Enter lowest value to test: 0
Enter highest value to test: 1000

0 1 2 3 4 5 6 7 8 9 153 370 371 407
*/
Finding Factorial Iteratively and Recursively
/*
The factorial of a non-negative integer 'n', denoted 
by n!, is the product of all positive integers less 
than or equal to n.
For example: 5! = 5 x 4 x 3 x 2 x 1 = 120
*/
```
```cpp
#include <iostream>
using namespace std;

int recursive_factorial(int n)
{
    if (n == 0) return 1;
    else return (n * recursive_factorial(n - 1));
}

int iterative_factorial(int n)
{
    int f = 1;
    for (int i = 1; i <= n; ++i)
        f *= i;

    return f;
}

int main()
{
    cout << "Find factorial of? ";
    int n; cin >> n;
    cout << "Using recursion, factorial is ";
    cout << recursive_factorial(n) << endl;
    cout << "Using iteration, factorial is ";
    cout << iterative_factorial(n) << endl;
    return 0;
}
```
```
/* OUTPUT

Find factorial of? 7
Using recursion, factorial is 5040
Using iteration, factorial is 5040
*/
```
Finding GCD Recursively

/*
The greatest common divisor (gcd) also known as the
greatest common factor (gcf) or highest common factor
(hcf) of two or more non-zero integers, is the largest
positive integer that divides the numbers without a 
remainder.

Steps:

1. Accept two values m and n, whose GCD we want to find  
2. Determine the smaller value between m and n and assign it to d  
3. Divide both m and n by d, if the remainder in both the cases is 0 then d is the required GCD, print the value and exit.  
4. Else if either of the division produces a non-zero remainder, decrement d  
5. Repeat step 3-4 until a GCD is found.  
*/
```cpp
#include <iostream>
using namespace std;

int gcd(int m, int n, int d = -1)
{
    if (d == -1)
        d = m > n ? n : m;

    if (m % d == 0 && n % d == 0)
        return d;

    else return gcd (m, n, d - 1);
}

int main()
{
    int m, n;
    cout << "Enter first number: ";
    cin >> m;
    cout << "Enter second number: ";
    cin >> n;
    cout << "GCD is " << gcd(m, n);
    return 0;
}
```
```
/* OUTPUT

Enter first number: 56
Enter second number: 42
GCD is 14
*/
```
Determining if a Number is Prime Iteratively 
/*
A prime number (or prime) is a natural number greater
than 1 that has no positive divisors other than 1 and
itself. A natural number greater than 1 that is not a
prime number is called a composite number.
Steps:

1. Get value n, which we want to test  
2. Let i = n - 1  
3. Divide n by i, if the remainder is non-zero the number is not prime, exit  
4. Repeat step 3 until we reach 1, if all division upto this point resulted in zero remainder, the number is prime.  
*/
```cpp
#include <iostream>
#include <cmath>
using namespace std;

int main()
{
    cout << "Program to test primality" << endl;
    cout << "Enter a number: ";
    int n = 0; cin >> n;

    // Loop from sqrt(n) to 1
    for (int i = sqrt(n); i > 0; --i)
    {
        if (i == 1)
        {
            cout << "Prime";
            break;
        }
        if ((n % i) == 0)
        {
            cout << "Not Prime";
            break;
        }
    }
    return 0;
}
```
```
/* OUTPUT

Program to test primality
Enter a number: 6
Not Prime
*/
```
Finding the Largest Palindromic Number Formed by Multiplying Two 3-Digit Numbers.
/*
Problem: Find the largest palindromic number formed by
multiplying two 3-digit numbers.

A palindromic number or numeral palindrome is a number
that remains the same when its digits are reversed.

Steps:

1. Start with largest value of palindrome as 0  
2. Make a reverse copy of the number to test  
3. Compare it with the original number value  
4. If the two are exactly equal, the number is a palindrome  
5. If the number is a palindrome and is larger than the current stored largest value, store the number as the new largest  
6. Repeat steps 2-5 for all combinations of i * j, where i and j range from 100-999  
7. The largest value at the end is the required largest value.  
*/
```cpp
#include <iostream>
using namespace std;
bool isPalindrome(int number)
{
    int original = number;
    int reverse = 0;

    while(number)
    {
        int remain = number % 10;
        number /= 10;
        reverse = reverse * 10 + remain;
    }
    return reverse == original;
}

----- OR -------

#include <iostream>
#include <sstream>
using namespace std;
bool isPalindromeNum(int number)
{
    stringstream ss(number);
    string original;
    ss >> original;
    for(int i=0,max=s.length()/2,len=s.length()-1;i<max;++i)
        if(s[i]!=s[len-i]) return false;
    return true;
}
-----------------------
int main()
{
    long largest = 0;

    for (int i = 999; i > 99; --i)
        for (int j = 999; j > 99; --j)
            if (isPalindrome(i * j) && i * j > largest)
                    largest = i * j;

    cout << "Largest palindrome is " << largest;
    return 0;
}
```
```
/* OUTPUT

Largest palindrome is 906609
*/
```
Finding the Largest Prime Factor of 600,851,475,143
/*
The prime factors of a positive integer are the prime
numbers that divide that integer exactly, without 
leaving a remainder. The process of finding these 
numbers is called integer factorization, or prime
factorization.
Steps:

1. Let n be the number whose prime factor is to be calculated  
2. Let s be the square root of n  
3. Divide n by s, if it leaves a remainder equal to zero, s is the prime factor, else continue to step 4  
4. Decrement by 1, repeat step 3 until a prime factor is found.  
*/
```cpp
#include <iostream>
#include <cmath>
using namespace std;

bool isPrime(long n, long i)
{
    if (i >= n) 
        return true;
    else if (n % i == 0) 
        return false;
    else return isPrime(n, ++i);
}

int main()
{
    long long n = 600851475143LL;
    long s = (long) sqrt(n);

    cout << "What is the largest prime factor "
         << "of the number 600851475143?" << endl;

    for (long i = s; i > 1; --i)
    {
        if ((n % i) == 0)
        {
            if (!isPrime(i, 2))
                continue;
            cout << "Answer: " << i;
            break;
        }
    }
    return 0;
}
```
```
/* OUTPUT

What is the largest prime factor of 600851475143?
Answer: 6857
*/
```
Finding the Transpose of a Matrix with Multidimensional Arrays
/*
Problem: To find the transpose of a given matrix.

The transpose of a matrix is formed by turning all the
rows of a given matrix into columns and vice-versa. The
transpose of matrix A is written A^T.

Steps:
1. Get the values for matrix A with 'r' rows and 'c' columns  
2. Copy value at rth row and cth column in matrix A to the cth row and rth column of result matrix (i.e. R[c][r] = A[r][c])  
3. Repeat step 2 for each element of matrix A  
*/
```cpp
#include <iostream>
using namespace std;

int main()
{
    int r = 0, c = 0;
    cout << "Enter the size of the matrix..." << endl;
    cout << "How many rows? ";
    cin >> r;
    cout << "How many columns? ";
    cin >> c;

    const int rows = r;
    const int cols = c;
       
    int** matrix = new int*[rows];
    int** result = new int*[cols];

    int i = 0, j = 0;

    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            matrix[r] = new int[cols];
            result[c] = new int[rows];
        }
    }

    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            cout << "Enter value for Matrix[" << r + 1
                 << "  " << c + 1 << "]: ";
            cin >> matrix[r][c];
        }
    }
 
    for (r = 0; r < cols; ++r)
        for (c = 0; c < rows; ++c)
            result[r][c] = matrix[c][r];

    cout << endl << "Original matrix is..." << endl;
    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
            cout << result[c][r] << " ";
        cout << endl;
    }

    cout << endl << "Result of transpose..." << endl;
    for (c = 0; c < cols; ++c)
    {
        for (r = 0; r < rows; ++r)
            cout << result[c][r] << " ";
        cout << endl;
    }

    // Clean up
    for (r = 0; r < rows; ++r)
    {
        delete [] matrix[r];
        delete [] result[r];
    }
    delete [] matrix;
    delete [] result;

    return 0;
}
```
```
/* OUTPUT

Enter the size of the matrix...

How many rows? 3
How many columns? 2
Enter value for Matrix[1  1]: 1
Enter value for Matrix[1  2]: 2
Enter value for Matrix[2  1]: 3
Enter value for Matrix[2  2]: 4
Enter value for Matrix[3  1]: 5
Enter value for Matrix[3  2]: 6

Original matrix is...
1 2
3 4
5 6

Result of transpose...
1 3 5
2 4 6
*/
```
Finding the Transpose of a Matrix with a Single Dimensional Array
/*
Problem: To find the transpose of a given matrix.

The transpose of a matrix is formed by turning all the
rows of a given matrix into columns and vice-versa. The
transpose of matrix A is written A^T.

Steps:

1. Get the values for matrix A with 'r' rows and 'c' columns  
2. Copy value at rth row and cth column in matrix A to the cth row and rth column of result matrix (i.e. R[c][r] = A[r][c])  
3. Repeat step 2 for each element of matrix A  
*/
```cpp
#include <iostream>
using namespace std;

int main()
{
    int r = 0, c = 0;
    cout << "Enter the size of the matrix..." << endl;
    cout << "How many rows? ";
    cin >> r;
    cout << "How many columns? ";
    cin >> c;

    const int rows = r;
    const int cols = c;

    // Use one large block of memory for the 2D array
    // matrix[i * cols + j] == matrix[i][j] 
    int* matrix = new int[rows * cols];
    int* result = new int[cols * rows];

    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            cout << "Enter value for Matrix[" << r + 1
                 << "  " << c + 1 << "]: ";
            cin >> matrix[r * cols + c];
        }
    }    

    for (r = 0; r < cols; ++r)
        for (c = 0; c < rows; ++c)
            result[r * cols + c] = matrix[c * rows + r];

    cout << endl << "Original matrix is..." << endl;
    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
            cout << result[c * rows + r] << " ";
        cout << endl;
    }

    cout << endl << "Result of transpose..." << endl;
    for (c = 0; c < cols; ++c)
    {
        for (r = 0; r < rows; ++r)
            cout << result[c * rows + r] << " ";
        cout << endl;
    }

    // Clean up
    delete [] matrix;
    delete [] result;
    return 0;
}
```
```
/* OUTPUT

Enter the size of the matrix...

How many rows? 3
How many columns? 2
Enter value for Matrix[1  1]: 1
Enter value for Matrix[1  2]: 2
Enter value for Matrix[2  1]: 3
Enter value for Matrix[2  2]: 4
Enter value for Matrix[3  1]: 5
Enter value for Matrix[3  2]: 6

Original matrix is...
1 2
3 4
5 6

Result of transpose...
1 3 5
2 4 6
*/
```
Iterative Binary Search
/*
Problem: Implement an Iterative Binary Search.

On average for finding any value in an unsorted array, 
complexity is proportional to the length of the array.

The situation changes significantly when the array is 
sorted. If we know it, random access capability can be
utilized very efficiently to quickly find the searched
value. The cost of the searching algorithm reduces to
binary logarithm of the array length. For reference,
log2(1000000) is approximately 20. It means that in the
worst case, the algorithm takes 20 steps to find a 
value in a sorted array of one million elements.

Steps:

1. Get the middle element  
2. If the middle element equals the searched value, the algorithm stops  
3. Otherwise, two cases are possible:  
    1. Searched value is less than the middle element. In this case go to step 1 for the part of the array before the middle element.  
    2. Searched value is greater than the middle element. In this case go to step 1 for the part of the array after the middle element.  
4. Iteration should stop when the searched element is found or when the sub-array has no elements. In the second case, we can conclude that the searched value is not present in the array.  
*/
```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int binarySearch(int haystack[], int needle, int length)
{
    int low = 0; 
    int high = length;
    int mid = (low + high) / 2;

    while (high >= low)
    {
        if (haystack[mid] == needle)
            return mid;
        else
        {
            if (needle > haystack[mid])
                low = mid + 1;
            else high = mid - 1;
            mid = (low + high) / 2;
        }
    }
    return -1; // not found
}

int main()
{    
    // Replace haystack with user input if needed
    // Haystack MUST BE SORTED
    int haystack[10] = {10, 11, 20, 22, 30,
                        33, 40, 44, 50, 55};
    int length = sizeof(haystack) / sizeof(int);
    srand(time(NULL));
    int needle = haystack[rand() % 10];

    cout << "This is the array: ";
    for (int i = 0; i < length; ++i)
        cout << haystack[i] << " ";

    cout << endl;
    cout << "Searching for " << needle << endl;
    cout << "Value is at array index: ";
    cout << binarySearch(haystack, needle, length);
    return 0;
}
```
```
/* OUTPUT

This is the array: 10 11 20 22 30 33 40 44 50 55
Searching for 40
Value is at array index: 6
*/
```
Removing Vowels from a String
```cpp
#include <iostream>
#include <string>
#include <cctype>
using namespace std;

const string vowels = "aeiou";

bool isVowel(char chr)
{
    for (int i = 0; i < 5; ++i)
        if (chr == vowels[i])
            return true;

    return false;
}

string removeVowels(string str)
{
    string finalString = "";
    int length = str.length();
    for (int i = 0; i < length; ++i)
        if (!isVowel(tolower(str[i])))
            finalString += str[i];
    
    return finalString;
}

int main()
{
    string str = "";
    cout << "Provide some text: ";
    getline(cin, str);
    cout << "The text you entered is: " 
         << str << endl;

    cout << "Your text with vowels removed "
         << "is: " << removeVowels(str) << endl;
    return 0;
}
```
```
/* OUTPUT

Please provide some text:
inky pinky ponky
The text you entered is: inky pinky ponky
Your text with vowels removed is: nky pnky pnky
*/
```
Finding the Roots of a Quadratic Equation
/*
Problem: Find the roots of a given quadratic equation.

A quadratic equation is an equation in the form ax^2 
+ bx + c = 0, where a is not equal to zero. The "roots"
of the quadratic are the numbers that satisfy the 
quadratic equation. There are always two roots for any
quadratic equation, although sometimes that may coincide.

The roots of any quadratic equation is given by: 
x = [-b +/- sqrt(-b^2 - 4ac)] / 2a

Steps:

1. Get the values of a, b, and c  
2. If a is equal to zero, the equation is not a quadratic equation  
3. Calculate the value of the discriminant: d = b^2 - (4ac)  
    1. If the discriminant is positive, then there are two distinct roots, both of which are real numbers.  
    2. If the discriminant is zero, then there is exactly one distinct real root, sometimes called a double root  
    3. If the discriminant is negative, then there are no real roots. Rather, there are two distinct (non-real) complex roots, which are complex conjugates of each other.  
*/
```cpp
#include <iostream>
#include <cmath>
using namespace std;

int main() 
{
    int a, b, c, d;
    double x1, x2;

    cout << "Program to find roots of quadratic equation" 
         << endl;
    cout << "Enter values for ax^2 + bx + c..." << endl;
    cout << "Enter value for a: ";
    cin >> a;
    cout << "Enter value for b: ";
    cin >> b;
    cout << "Enter value for c: ";
    cin >> c;
    cout << endl;

    if (a == 0)
        cout << "Not a quadratic equation.";

    else 
    {
        d = (b * b) - (4 * a * c);
        if (d > 0)
        {
            cout << "Real and distinct roots" << endl;
            x1 = ((-b + sqrt(d)) / (2 * a));
            x2 = ((-b - sqrt(d)) / (2 * a));

            cout << "Root 1 = " << x1 << endl;
            cout << "Root 2 = " << x2 << endl;
        }
        else if (d == 0)
        {
            cout << "Real and equal roots" << endl;
            x1 = x2 = -b / (2 * a);
            cout << "Root 1 = " << x1 << endl;
            cout << "Root 2 = " << x2 << endl;
        }
        else {
            cout << "Imaginary roots" << endl;
            x1 = -b / (2 * a);
            x2 = sqrt(-d) / (2 * a);
            cout << "Root 1 = " << x1 << endl;
            cout << "Root 2 = " << x2 << endl;
        }
    }
    return 0;
}
```
```
/* OUTPUT

Program to find roots of quadratic equation
Enter values for ax^2 + bx + c...
Enter value for a: 1
Enter value for b: -2
Enter value for c: -15

Real and distinct roots
Root 1 = 5
Root 2 = -3
*/
```
Finding the Sum of Even-Valued Terms in the Fibonacci Sequence Less than 4 Million
```cpp
#include <iostream>
using namespace std;

int main() 
{
    cout << "By considering the terms in the Fibonacci "
         << "sequence whose values do not exceed four "
         << "million, find the sum of the even-valued "
         << "terms." << endl;

    int x = 0, y = 1, t = 0, sum = 0;
    while (y < 4000000)
    {
        if (y % 2 == 0) sum += y;

        t = x + y;
        x = y;
        y = t;
    }
    cout << "Answer: " << sum;
    return 0;
}
```
```
/* OUTPUT

Answer: 4613732
*/
```
Finding the Sum of Even Element Values in a Matrix
```cpp
#include <iostream>
using namespace std;

int main()
{ 
    const int size = 4;
    int sum = 0;

    int matrix[size][size] = {{10, 11, 20, 22},
                              {30, 33, 40, 44},
                              {50, 55, 60, 66},
                              {70, 77, 80, 88}};

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            cout << matrix[i][j] << " ";
            if (matrix[i][j] % 2 == 0)
                sum += matrix[i][j];
        }
        cout << endl;
    }
    cout << "Sum of evens: " << sum << endl;
    return 0;
}
```
```
/* OUTPUT

10 11 20 22
30 33 40 44
50 55 60 66
70 77 80 88
Sum of evens: 580
*/
```
The Tower of Hanoi
/* 
The Tower of Hanoi is a mathematical game or puzzle. It
consists of three rods, and a number of disks of different
sizes which can slide onto any rod. The puzzle starts with
the disks in a neat stack in ascending order of size on one
rod, the smallest at the top, thus making a conical shape.

The object of the puzzle is to move the entire stack to 
another rod, obeying the following rules:
1. Only one disk may be moved at a time  
2. Each move consists of taking the upper disk from one of the rods and sliding it onto another rod, on top of the other disks that may already be present on that rod.  
3. No disk may be placed on top of a smaller disk.  

A key to solving this puzzle is to recognize that it can be
solved by breaking the problem down into a collection of
smaller problems and further breaking those problems down 
into even smaller problems until a solution is reached.
The following procedure demonstrates this approach.
    * label the pegs A, B, C -- these labels may move at
      different steps
    * let n be the total number of discs
    * number the discs from 1 (smallest, topmost) to n
      (largest, bottommost)

To move n discs from peg A to peg C:
1. move n - 1 discs from A to B. This leaves disc n
       alone on peg A
2. move disc n from A to C
3. move n - 1 discs from B to C so they sit on disc n

The above is a recursive algorithm: to cary out steps 1 and
3, apply the same algorithm again for n - 1. The entire
procedure is a finite number of steps, since at some point
the algorithm will be required for n = 1. This step, moving 
a single disc from peg A to peg B, is trivial.
*/
```cpp
#include <iostream>
using namespace std;

void hanoi(int n);
void moveTower(int ht, char f, char t, char i);
void moveRing(int d, char f, char t);

int main()
{
    cout << "How many disks? ";
    int x; cin >> x;
    hanoi(x);
    return 0;
}

void hanoi(int n) 
{
    moveTower(n, 'A', 'B', 'C');
}

void moveTower(int ht, char f, char t, char i)
{
    if (ht > 0)
    {
        moveTower(ht - 1, f, i, t);
        moveRing(ht, f, t);
        moveTower(ht - 1, i, t, f);
    }
}

void moveRing(int d, char f, char t)
{
    cout << "Move ring " << d << " from ";
    cout << f << " to " << t << endl;
}
```
```
/* OUTPUT

How many disks? 3
Move ring 1 from A to B
Move ring 2 from A to C
Move ring 1 from B to C
Move ring 3 from A to B
Move ring 1 from C to A
Move ring 2 from C to B
Move ring 1 from A to B
*/
```
Kruskal’s Algorithm
/*
Kruskal's algorithm is an algorithm in graph theory that
finds a minimum spanning tree for a connected weighted
graph. This means it finds a subset of the edges that forms
a tree that includes every vertex, where the total weight of
all the edges in the tree is minimized. If the graph is not 
connected, then it finds a minimum spanning forest (a
minimum spanning tree for each connected component).
Kruskal's algorithm is an example of a greedy algorithm.

Steps:

1. Create a forest F (a set of trees), where each vertex in the graph is a separate tree  
2. Create a set S containing all the edges in the graph  
3. While S is nonempty and F is not yet spanning  
    1. Remove an edge with minimum weight from S  
    2. If that edge connects two different trees, then add it to the forest, combining two trees into a single tree  
    3. Otherwise discard that edge  

At the termination of the algorithm, the forest has only one
component and forms a minimum spanning tree of the graph.
*/
```cpp
#include <iostream>
#include <algorithm>
using namespace std;

struct Edge 
{
    int m_first_vertex, m_second_vertex, m_weight;
};

bool checkCycle (Edge e, int path[]);

int main()
{
    // Create graph of 'n' vertices and 'm' edges
    cout << "Enter the number of vertices in the graph: ";
    int n = 0; cin >> n;    
    cout << "Enter the number of edges in the graph: ";
    int m = 0; cin >> m;

    int path[n + 1];
    struct Edge e[m];

    int i;
    cout << endl;
    for (i = 0; i < m; ++i)
    {
        cout << "Enter 2 vertices and weight of edge " 
             << i + 1 << endl;
        cout << "First vertex: ";
        cin >> e[i].m_first_vertex;
        cout << "Second vertex: ";
        cin >> e[i].m_second_vertex;
        cout << "Weight: ";
        cin >> e[i].m_weight;
        cout << endl;
    }

    // Sort the edges in ascending order of weights
    int j;
    for (i = 0; i <= (m - 1); ++i)
        for (j = 0; j < (m - i - 1); ++j)
            if (e[j].m_weight > e[j + 1].m_weight)
                swap (e[j], e[j + 1]);

    // Initialize the path array
    for (i = 1; i <= n; ++i)
        path[i] = 0;

    // Counts the number of edges selected in the tree
    i = 0;

    // Counts the number of edges selected or discarded
    j = 0;

    int minimum_cost = 0;

    while ((i != (n - 1)) && (j != m))
    {
        cout << "Edge (" << e[j].m_first_vertex << ", " 
             << e[j].m_second_vertex << ") with weight "
             << e[j].m_weight << " ";

        if (checkCycle (e[j], path))
        {
            minimum_cost += e[j].m_weight;
            i++;
            cout << "is selected";
        } 

        else cout << "is discarded";
        cout << endl;
        j++;
    }

    if (i != (n - 1))
        cout << "Minimum spanning tree cannot be formed";

    return 0;
}

bool checkCycle (Edge e, int path[])
{
    int first_vertex = e.m_first_vertex;
    int second_vertex = e.m_second_vertex;

    while (path[first_vertex] > 0) 
        first_vertex = path[first_vertex];

    while (path[second_vertex] > 0)
        second_vertex = path[second_vertex];

    if (first_vertex != second_vertex)
    {
        path[first_vertex] = second_vertex;
        return true;
    }
    return false;
}
```
```
/* OUTPUT

Enter the number of vertices in the graph: 4
Enter the number of edges in the graph: 4

Enter 2 vertices and weight of edge 1
First vertex: 1
Second vertex: 2
Weight: 6

Enter 2 vertices and weight of edge 2
First vertex: 1
Second vertex: 4
Weight: 5

Enter 2 vertices and weight of edge 3
First vertex: 1
Second vertex: 3
Weight: 7

Enter 2 vertices and weight of edge 4
First vertex: 3
Second vertex: 4
Weight 8

Edge (1, 4) with weight 5 is selected
Edge (1, 2) with weight 6 is selected
Edge (1, 3) with weight 7 is selected
*/
```
Finding the Angle between the Hour Hand and the Minute Hand on an Analog Clock
```cpp
#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

double angle(int hour, int minute)
{
    // Hour angle (from 12 o'clock): 360 * hours/12
    // ==> the hour hand moves at the rate of 30 degrees per hour
    //      or 0.5 degrees per minute
    double hour_angle = 0.5 * (hour * 60 + minute);

    // Minute angle (from 12 o'clock): 360 * minutes/60
    // ==> minute hand moves at the rate of 6 degrees per minute
    double minute_angle = 6 * minute;

    double angle_between = abs(hour_angle - minute_angle);
    angle_between = min(angle_between, 360 - angle_between);
    return angle_between;
}

int main() 
{
    cout << "Enter hour (1 - 12): ";
    int hour = 1; cin >> hour;
    cout << "Enter minute (0 - 59): ";
    int minute = 0; cin >> minute;
    cout << "Angle between hour hand and minute hand: "
         << angle(hour, minute) << " degrees" << endl;
    return 0;
}
```
```
/* VARIOUS OUTPUTS

Enter hour (1 - 12): 2
Enter minute (0 - 59): 20
Angle between hour hand and minute hand: 50 degrees

Enter hour (1 - 12): 3
Enter minute (0 - 59): 15
Angle between hour hand and minute hand: 7.5 degrees

Enter hour (1 - 12): 1
Enter minute (0 - 59): 0
Angle between hour hand and minute hand: 30 degrees

Enter hour (1 - 12): 6
Enter minute (0 - 59): 0
Angle between hour hand and minute hand: 180 degrees
*/
```
What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
```cpp
#include <iostream>
#include <ctime>
using namespace std;

int main() 
{
    int n, i, t;
    bool found = false;

    for (n = 1; true; ++n)
    {
        found = true;
        for (int i = 1; i <= 20; ++i)
        {
            if (n % i != 0) 
            {
                found = false;
                break;
            }
        }
        if (found) break;
    }
    cout << n << endl;
    
    t = clock();
    cout << "The operation took " 
         << static_cast<double>(t)/CLOCKS_PER_SEC
         << " seconds." << endl;
    return 0;
}
```
```
/* OUTPUT

232792560
The operation took 3.28 seconds.
*/
```
Samples
Hello World
```cpp
#include <iostream>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
cout << “Hello World” << endl;
return 0;
}
```
Center Point of Arbitrary Points
```cpp
double centerX = sum(AllXPoints[])
double centerY = sum(AllYPoints[])
```

Numeric - bool isdigit(char)
Alphanumeric - bool isalnum(char)

Input Handling

Includes
```cpp
#include<sstream>
```
String Stream
```cpp
#include<algorithm>
```
remove_if
Helpful Functions
Character Checking
Check if a character is
Alphabetic - `bool isalpha(char)`
Numeric - `bool isdigit(char)`
Alphanumeric - `bool isalnum(char)`

Programming tricks
Compile and Run (one line)
```
g++ myFile.cpp && ./a.out
```
Use file for stdin
```
./a.out < file.in
```
Send stdout to file
```
./a.out > file.out
```


Misc.
*nix - Time Command
To use the command, simply precede any command by the word time, such as:
time ls
When the command completes, time will report how long it took to execute the ls command in terms of user CPU time, system CPU time, and real time. The output format varies between different versions of the command, and some give additional statistics
$ time host wikipedia.org
wikipedia.org has address 207.142.131.235
0.000u 0.000s 0:00.17 0.0% 0+0k 0+0io 0pf+0w
$
For more info, type “man time”





Map-of-maps iteration
```cpp
#include<map>
#include<utility>

std::pair <std::string,double> product1; // default constructor
std::pair <std::string,double> product2 ("tomatoes",2.30); // value init
std::pair <std::string,double> product3 (product2); // copy constructor


map<string, innerMap >::iterator it;
map<string,int >::iterator inner_it;
it=m.find(target);
for( inner_it=(*it).second.begin(); inner_it != (*it).second.end(); inner_it++)
{
    myQ.update_weight((*inner_it).first,(*it).first,(*inner_it).second);
}
```
## Graph Algorithms
### All-pairs shortest path algorithms
#### Floyd-Warshall O(n<sup>3</sup>)

##### Pseudocode
```
#setup
for i = 1 to N
   for j = 1 to N
      if there is an edge from i to j
         dist[0][i][j] = the length of the edge from i to j
      else
         dist[0][i][j] = INFINITY
#computation
for k = 1 to N
   for i = 1 to N
      for j = 1 to N
         dist[k][i][j] = min(dist[k-1][i][j], dist[k-1][i][k] + dist[k-1][k][j])

```

##### C++
```cpp

/**
 * Reads input from stdin into a matrix
 *
 * This expects the input to be formatted like so for a 3x3 matrix:
 * 1 2 3
 * 4 5 6
 * 7 8 9
 *
 * @param int n the size of our matrix (square matrix only for a graph)
 * @param vector< vector<int> > &m a matrix (2d vector)
 */
void read_input_to_matrix(int n, vector< vector<int> > &m) {
    int tmp;
    for (int i = 0; i < n; i++) {
        vector<int> ivec;
        for (int j = 0; j < n; j++) {
            cin >> tmp;
            ivec.push_back(tmp);
        }
        m.push_back(ivec);
    }
}

/**
 * Computes the all-pairs shortest path using Floyd-Warshall and updates our matrix
 * @param m the matrix of flight times
 */
void shortest_paths(vector< vector<int> > &m) {
    int n = m.size();
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (m[i][j] > m[i][k] + p[k] + m[k][j]) {
                    m[i][j] = m[i][k] + p[k] + m[k][j];
                }
            }
        }
    }
}
```