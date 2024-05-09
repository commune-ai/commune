s1 = 'I want to read these texts side by side and see how similar they really are'
s2 = 'I really want to read these here texts side by side to see how similar they are (qualitatively)'

maxChars = 40
maxLength = max(len(s1),len(s2))

s1 = s1.ljust(maxLength," ")
s2 = s2.ljust(maxLength," ")

s1 = [s1[i:i+maxChars] for i in range(0,len(s1),maxChars)]
s2 = [s2[i:i+maxChars] for i in range(0,len(s2),maxChars)]
import commune as c
for elem1, elem2 in zip(s1,s2):
    c.sleep(1)
    print(elem1.ljust(maxChars," "), end="    ")
    print(elem2)