# Coding Trees

Only one of the folowing tags may be pressent when you run the program 
```console
-encode : single flag tells the program to encode required flags for this are
         : -input or -inputF (if both are present the later flag takes precident)
         : -enocdeDict or -encodeDictF (same as above)

-decode : single flag tells the program to encode required flags for this are
         : -input or -inputF (if both are present the later flag takes precident)
         : -enocdeDict or -encodeDictF (same as above)

-generate : single flag generates the dictionary from a hoffman encoding required flags are 
           : -input or -inputF (if both are present the later flag takes precident)
           : -enocdeTokens or -tokensF (same as above)
```

The following are the remaining flags and what they do
```console
-file: if present the output will be printed to a file named output.txt into the current working directory

-input: takes the string following it as the input you must use string encapsulation in yout prefered console emulator

-inputF: takes a file name as the next argument based on the cwd as input

-enocdeTokens: takes a string in the next argument encapsulated by whatever command line utility of tokens to use for generating a encoing map this list should be space seperated as such
             : "0 1 2 3 4 5 6 7 8 9 a b c d e f"

-tokensF: takes a file name as the next argument based on the cwd of a line break seperated list of tokens

-encodeDict: takes a dictionary as a string (encapsulated) from the next arg this list should be space seprated
           : " =10 d=01"

-encodeDictF: takes a file directory based on cwd of a dict with entries enter sperated
```


## encode tokens file example
```text
0
1
2
3
4
5
6
7
8
9
a
b
c
d
e
f
```
## encode/decode mapping dict file example
```text
 =110
a=0001
b=00100
r=00110
s=10
t=00111
e=111
h=010
y=0000
l=011
o=00101
```

## example runs

```console
$ java -jar HW04-CodingTrees.jar -encode -inputF test.txt -encodeDictF decode.txt 
100101111101011101101110110101110001110100101110110111011000100000011000111010111110101110001110100100010100110111
```
---
```console
$ java -jar HW04-CodingTrees.jar -generate -input "she sells sea shells by the sea shore"  -encodeTokens "0 1 2 3 4 5 6 7 8 9" -file
```

output.txt
```text
{ =18, a=15, b=10, r=12, s=0, t=13, e=19, h=16, y=14, l=17, o=11}
```