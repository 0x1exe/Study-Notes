
# Backtracking
## Permutations 

**Non-unique** permutations: 
```
def permute(nums):
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            path.append(nums[i])
            used[i] = True
            backtrack(path, used)
            path.pop()
            used[i] = False

  
    result = []
    used = [False] * len(nums)
    backtrack([], used)
    return result
```

Faster:
```
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        # base case
        if len(nums) == 1:
            return [nums[:]]  # nums[:] is a deep copy
        for i in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)
            for perm in perms:
                perm.append(n)
            res.extend(perms)
            nums.append(n)
        return res

```
Even faster:
```
class Solution: 
def permute(self, nums: List[int]) -> List[List[int]]: 
	res = []
	s = set(nums) 
	def dfs(curr, left): 
		if len(left) == 0: 
			res.append(curr) 
		for n in left: 
			dfs(curr+[n], left-{n}) 
	dfs([], s) 
	return res
```


**Unique** permutations:
```
def unique_permutations(nums):
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i-1] and not used[i-1]):
                continue
            
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    nums.sort()
    result = []
    used = [False] * len(nums)
    backtrack([], used)
    return result
    
nums = [1, 1, 2]
permutations = unique_permutations(nums)
for perm in permutations:
    print(perm)


```

## Combinations
```
def combine(nums, k):
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    result = []
    backtrack(0, [])
    return result
```

**Unique** combinations
```
def unique_combinations(nums, k):
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    nums.sort()
    result = []
    backtrack(0, [])
    return result
```

## Solutions

### Basic subsets
No need for return condition, just add subsets to the result'
### combinationSum
We need to keep track of the target and it's difference with the current element. Do not move the element until the end root is reached (target negative)

Another solution is to avoid looping all together by keeping track of i,cur,tota;

```
def dfs(i, cur, total):
            if total == target:
                res.append(cur.copy())
                return
            if i >= len(candidates) or total > target:
                return
            cur.append(candidates[i])
            dfs(i, cur, total + candidates[i])
            cur.pop()
            dfs(i + 1, cur, total)
        dfs(0, [], 0)
```

### phone combinations

```
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
      keyboard = {"2": "abc","3": "def","4": "ghi","5": "jkl","6": "mno","7": "pqrs","8": "tuv","9": "wxyz"}
      result = []
      def backtrack(start,curr):
        if len(curr) == len(digits):
          result.append(curr[:])
          return
        for c in keyboard[digits[start]]:
          backtrack(start+1,curr+c)
      backtrack(0,'')
      return result
```

### CombinationSum3

Sometime adding an appropriate stoping condition leads to drastic speed increase:

```
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
      numbers = [1,2,3,4,5,6,7,8,9]
      result = []
      def backtrack(start,curr,total):
        if len(curr) == k and total == 0:
          result.append(curr[:])
          return
        for i in range(start,len(numbers)):
          if numbers[i] > n:   ----------------------> 50% slower without this
            break
          curr.append(numbers[i])
          backtrack(i+1,curr,total-numbers[i])
          curr.pop()
      backtrack(0,[],n)
      return result
```

### Additive numbers
```
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        n = len(num)
        # check if the sequence is valid starting from the first two numbers
        for i in range(1, n):
            for j in range(i+1, n):
                # if the first two numbers have leading zeros, move on to the next iteration
                if num[0] == "0" and i > 1:
                    break
                if num[i] == "0" and j > i+1:
                    break
                 
                # initialize the first two numbers and check if the sequence is valid
                num1 = int(num[:i])
                num2 = int(num[i:j])
                k = j
                while k < n:
                    # calculate the next number in the sequence and check if it matches the remaining string
                    num3 = num1 + num2
                    if num[k:].startswith(str(num3)):
                        k += len(str(num3))
                        num1 = num2
                        num2 = num3
                    else:
                        break
                if k == n:
                    return True
                
        # if no valid sequence is found, return False
        return False
```
```