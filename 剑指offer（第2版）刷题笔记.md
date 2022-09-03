## **LeetCode 《剑指offer（第2版）》 刷题笔记**

这里的题目都是我自己一刷时没do出来的（看题解） or do的有些磕磕绊绊需要想挺久才do出来的题目！！！（暗示我自己这些剑指offer的题目为肯定是需要2刷甚至3or4刷的！）

我到时候面试的时候就一定要先把==10大排序算法==再背回来！清除知道了解他们的时间空间复杂度！==(這一點是我在刷lc题目的时候没注意到的！)==

### 目前累计总共有==《29》==道题：

#### <1> [LeetCode 剑指offer 04.二维数组中的查找：](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

题解：

```c++
// 学习题解区的高赞题解！
// 来自B站Up主:香辣鸡排蛋包饭 小姐姐的视频！
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int i = matrix.size()-1,j = 0;
        while(i >= 0 && j < matrix[0].size()){
            if(matrix[i][j] < target)j++;
            else if(matrix[i][j] > target)i--;
            else return true;
        }
        return false;
    }
};
```

#### <2> [LeetCode 剑指offer 25.合并两个排序的链表：](https://leetcode.cn/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

题解：

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* newList = new ListNode(-1);
        ListNode* cur = newList;
        // 这道题目就是整体上思路需要理清楚才能够一次性写出来无wrong的codes!
            /*  整体思路：
                case1:l1空 且 l2空 == 直接break跳出构建新list的while循环
                case2:l1不空 且 l2空 == 先把l1剩余的all元素拿来构建新list然后再break
                case3:l1空 且 l2不空 == 先把l2剩余的all元素拿来构建新list然后再break
                case4:l1不空 且 l2不空
                    case5: l1->val < l2->val == 先把l1当前的元素拿来构建新list然后再continue
                    case6: l1->val >= l2->val == 先把l2当前的元素拿来构建新list然后再continue
            */ 
        while(true){
            if(!l1 && !l2) break;
            else if(l1 && !l2){
                while(l1){
                    ListNode* newNode = new ListNode(l1->val);
                    cur->next = newNode;
                    cur = cur->next;
                    l1 = l1->next;
                }
                break;
            }else if(!l1 && l2){
                while(l2){
                    ListNode* newNode = new ListNode(l2->val);
                    cur->next = newNode;
                    cur = cur->next;
                    l2 = l2->next;
                }
                break;
            }
            else if(l1 && l2){
                ListNode* newNode = new ListNode();
                if(l1->val < l2->val){
                    newNode->val = l1->val;
                    cur->next = newNode;
                    l1 = l1->next;
                }else{
                    newNode->val = l2->val;
                    cur->next = newNode;
                    l2 = l2->next;
                }
            }
            cur = cur->next;
        }
        return newList->next;
    }
};
```

#### <3> [LeetCode 剑指offer 31.栈的压入、弹出序列：](https://leetcode.cn/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

（这个题目我想到了lc高赞答案的题解了，但是没有自己写出来！）

```c++
// time:O(n) 一个for
// space:O(n) st最大可以n个元素
/*
	总体思路：借助一个stack来do，模拟压栈和出栈的case即可！
	压栈就按照pushed数组的顺序压入即可，然后同时将st.top()头部元素与popped元素依次顺序do比较
	if st.top()头部元素与popped的当前元素
		st.pop()并且让指向popped的下标后移
	else 继续遍历pushed数组并将其元素push进st中
	最终判断st是否为空即可判断是否为正确的压栈出栈操作序列了！
	if st.empty() == true ==> return true
	if st.empty() == false ==> return false
*/
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        // 拉个辅助stack即可了！
        stack<int> st;
        int idx_poped = 0;
        for(int i = 0;i < pushed.size();++i){
            st.push(pushed[i]);
            while(!st.empty() && st.top() == popped[idx_poped]){
                idx_poped++;
                st.pop();
            }
        }
        return st.empty();
    }
};
```



#### <4> [LeetCode 剑指offer 50.第一个只出现一次的字符：](https://leetcode.cn/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

```c++
// 暴力法的话就太tmd简单了（这里就不写了，也就2层for循环就可以暴力出来了！）！
/*
    思路：
    	step1: 遍历字符串s，先用哈希表来记录s中哪些字符是只会出现一次的，是的就true，否的就false。
    	step2: 再遍历一次字符串s，看哈希表中记录着的哪个只出现一次的字符（即it->second == true的字符）处在s中的最前面，谁处在最前面就代表着是s字符串中的第一个只出现一次的字符了。
*/
// 因为hashMap只会存储不重复key的pair对，又因为s字符串只有小写字母，
// 因此只会存在26种可能，so空间复杂度为O(26) == O(1)
// 时间复杂度上，遍历s是O(2*n) == O(n),n是s字符串的长度,而map的操作的时间复杂度只是O(1)而已！
// time:O(n)
// space:O(1)
class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<char,bool> hashMap;
        for(char& ch : s){
            auto it = hashMap.find(ch);
            if(it != hashMap.end())it->second = false;
            else hashMap[ch] = true;
            // 一句代码也可以这么写：
            // hashMap[ch] = hashMap.find(ch) == hashMap.end();
        }
        for(char& ch : s){
            // 字符串s中谁先出现一次就返回谁！
            if(hashMap[ch])return ch;
        }
        return ' ';
    }
};
```



#### <5> [LeetCode 剑指offer 35.复杂链表的复制：](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

这是lc剑指offer板块中的==评论区高赞大佬Krahets==教我的题解：

![lc剑指offer35](F:\学习过程中的截图\lc剑指offer35.png)

```c++
// time:O(n)
// space:O(n)
/* 整体思路：
            分2步走：
            1-new出新list的all节点(因为是复制copy，那肯定要new出来的！)并用hashMap记录new的all新节点
            2-利用前面hashMap记录new的all新节点，拿这些节点来do新的list连线工作（即上图中的虚线部分，）！
        */
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head == nullptr)return head;
        unordered_map<Node*,Node*> hashMap;
        // key:存放的是旧的list节点
        // val:存放的是新的list节点
        Node* cur = head;
        while(cur){// 这一步只是new处全新的list的节点而已！
            hashMap[cur] = new Node(cur->val);
            cur = cur->next;
        }
        cur = head;
        while(cur){// 这一步给这个新的list的各个节点连上线！
        // 包括连上next和random的线！
            hashMap[cur]->next = hashMap[cur->next];
            hashMap[cur]->random = hashMap[cur->random];
            cur = cur->next; 
        }
        return hashMap[head];
    }
};
```

#### <6> [LeetCode 剑指offer 54.二叉搜索树的第k大节点：](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
 /*解题思路：
 	已知：二叉搜索树的中序遍历是递增序列(这可是Data Structure中的常识)
 	那么求二叉搜索树的第k大节点的值 <==> 求其中序遍历的逆序（递减序列）的第k个节点的值！
  	你拿lc官方给出的二叉搜索树画一画举个例子就知道了哈！
 */ 
class Solution {
public:
    vector<int> tmp;
    void reverse_inOrder(TreeNode* cur){
        if(cur == nullptr)return;
        // 右 中 左
        reverse_inOrder(cur->right);
        tmp.push_back(cur->val);
        reverse_inOrder(cur->left);
        return;
    }
    int kthLargest(TreeNode* root, int k) {
        tmp.clear();
        reverse_inOrder(root);
        return tmp[k-1];// 下标是从0开始，而第几个元素时从1开始计数的！so要减去1！
    }
};
```



#### <7> [LeetCode 剑指offer 55II.平衡二叉树：](https://leetcode.cn/problems/ping-heng-er-cha-shu-lcof/)

这道题目为在carl哥的网站上刷了N次了，但是刷剑指offer时候还是有点陌生，==中途遍历到发现左右子树是非平衡就直接返回-1表示整棵树是非平衡这2行代码为没有写出来！！！==

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    // 判断一棵BT是否为一棵二叉平衡树
    // 求BT的深度 == BT的前序遍历的logic了！
    // 但是这个题目貌似用后续遍历的方式更加容易理解！
    // 用int值-1表示该BT是非平衡二叉树！
    int traversal(TreeNode* cur){
        if(cur == nullptr)return 0;
        int left = traversal(cur->left);
        if(left == -1)return -1;// 若中途遍历BT的时候发现有非平衡的case也要马上返回！
        int right = traversal(cur->right);
        if(right == -1)return -1;// 若中途遍历BT的时候发现有非平衡的case也要马上返回！
        if(abs(left - right) > 1)return -1;
        int mid = max(left,right)+1;
        return mid;// 返回当前BT的中节点为根的BT的深度！
    }
    bool isBalanced(TreeNode* root) {
        if(root == nullptr)return true;// 空树也属于平衡树的一种！
        return traversal(root) != -1;
    }
};
```



#### <8> [LeetCode 剑指offer 56I.数组中数字出现的次数](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)



```c++
// time:O(n)
// spcae:O(1)
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        // If用哈希表法，则是双O(n)的算法！因为hashMap需要存储nums.size()个元组对！
        // so我自己想了一种办法！
        // 先sort升序排序方便找出重复元素然后一个while循环就deal问题了！
        // 可以拿[1 1 4 6] or [1 1 2 3 3 4 4 10] 来画图举例子验证我写的codes！
        sort(nums.begin(),nums.end());// sort后可以方便找出重复的元素！   
        int ret1 = -1,ret2 = -1;
        int i = 0,size = nums.size();
        while(i < size - 1){
            if(nums[i] == nums[i+1])i = i+1+1;
            else {
                if(ret1 == -1){
                    ret1 = nums[i];
                    i++;
                }else{
                    ret2 = nums[i];
                    i++;
                }
            }
        }
        // 处理一下末尾位置index的尴尬情况！
        if(nums[size-1] != nums[size-2])ret2 = nums[size-1];
        // 不论是 [1 1 4 6] or [1 1 2 3 3 4 4 10] 
        // 这2种case，都让i遍历到size-1这个位置就直接break出循环没法继续遍历了
        // 因此任何情况下其实nums[size-1]这个位置的元素并没有判断是否是不重复元素
        // 因此为这里需要再判断下！
        return {ret1,ret2};
    }
};
```

==LRU算法（大厂面试很喜欢出的一个题目！！！）：==

#### <9> [LeetCode 剑指 Offer II 031. 最近最少使用缓存](https://leetcode.cn/problems/OrIXps/)

这个思路为是跟着b站一个小up主学的
网站：https://www.bilibili.com/video/BV18A411i7ui?spm_id_from=333.337.search-card.all.click&vd_source=b050ab0adaffa51f5ad24d77efa40057

```c++
// 今晚我就把这个LRL算法彻彻底底学会即可了
// 因为这是个非常常见常考的面试题目！
// 先来默写个100遍！
#include<iostream>
#include<unordered_map>
using namespace std;
// 首先，LRU要求删除添加和查找的速度都快
// 因此就必须要结合双向list和哈希表这2种数据结果来do了！
int main(int argc, char* argv[]) {
	return 0;
}
// 首先，LRU要求删除添加和查找的速度都快
// 因此就必须要结合双向list和哈希表这2种数据结果来do了！

// 节点-结构体
struct Node {
	int _key, _val;   // 键 - 值
	Node* prev;		  // 指当前元素的 前一个元素 的指针 
	Node* next;		  // 指当前元素的 后一个元素 的指针 
	Node(int k, int v) // 构造函数
		:_key(k), _val(v), prev(nullptr), next(nullptr) {}
};
// 双向链表list（好do删除节点操作，若用单向list就麻烦很多，但也能do）
// 作用：存储数据
/*
	case1：正常存入（存入时容量未满）
	case2：存入的是重复的key
	case3：存入时容量满了
*/
struct DoubleList {
private:
	Node* head;					// 头结点处放入 最近刚刚使用过的 元素（头插法） 
	Node* tail;					// 尾结点处删除 最久没被使用过的 元素（尾删法）
public:
	DoubleList() {
		head = new Node(0, 0);
		tail = new Node(0, 0);
		head->next = tail;
		tail->prev = head;
	}
	~DoubleList() {
		if (head) {
			delete head; head = nullptr;
		}
		if (tail) {
			delete tail; tail = nullptr;
		}
	}
	void addFirst(Node* node) { // 在list头部插入元素
		node->next = head->next;
		node->prev = head;
		node->next->prev = node;
		head->next = node;
	} 
	int remove(Node* node) {// 容量满了之后直接删除最后一个节点即可
		int res = node->_key;
		node->next->prev = node->prev;
		node->prev->next = node->next;
		delete node; node = nullptr;
		return res;
	}	// 在list尾部删除元素
	int removeLastNode() {// 容量满了之后直接删除最后一个节点即可
		if (head->next == tail)return -1;
		return remove(tail->prev);
	}		
};
class LRUCache {
private:
	int _capacity;
	DoubleList* dlist;
	unordered_map<int, Node*> map;
public:
	LRUCache(int cap) {
		this->_capacity = cap;
		dlist = new DoubleList();
	}
	~LRUCache() {
		if (dlist) {
			delete dlist; dlist = nullptr;
		}
	}
	/* get方法分为2种case：
		1-map中	没有key对应的pair
		2-map中	有key对应的pair
	*/
	// get方法：获取key对应的value
	int get(int key) {
		auto it = map.find(key);
		if (it == map.end())return -1;	// map中没有
		// 有,就do 2个步骤：
		// 1-先将pair对的value拿出来，2-然后放到队列头部（因为最近使用过了）！
		int res = it->second->_val;
		put(key, res);			// 头插回去LRU中
		return res;
	}
	// put方法：将最近使用过的或者新的pair<key,val>插入到LRU的头部！
	void put(int key, int val) {
		// 找到
		// 找不到
		Node* tnode = new Node(key, val);
		auto it = map.find(key);
		if (it != map.end()) {// 找到有重复key时，直接删除，无需从尾部删除！
			dlist->remove(it->second);	// 先删除双向list中的该节点（）
			dlist->addFirst(tnode);		// 后再向双向list的头部添加该节点
			map[key] = tnode;			// 再重新do哈希表的映射关系
		}
		else {
			if (map.size() == this->_capacity) {// 因为满了so只能从尾部进行删除了！
				int nodeKey = dlist->removeLastNode();// 容量满了就直接删除最后一个元素
				// 然后把映射关系也删除！
				map.erase(nodeKey);// 按照key值把map中保存的映射关系给删除掉！
				// 然后再从头部添加进去！
				dlist->addFirst(tnode);
				map[key] = tnode;// 再do个哈希表的映射关系
			}
			else if (map.size() < this->_capacity) {
				dlist->addFirst(tnode);// 若没有达到容量,直接从头部添加
				map[key] = tnode;// 再do个哈希表的映射关系
			}
		}
	}
};
```



#### <10> [LeetCode 剑指 Offer 66.构建乘积数组](https://leetcode.cn/problems/gou-jian-cheng-ji-shu-zu-lcof/)

```c++
/*
    time:O(n * 2) == O(n)
    space:O(n),if不算res返回数组的话那就是O(1)的空间复杂度！
*/
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        int size = a.size();
        vector<int> res(size,1);
        // 本题为通过学习评论区高赞答案：通过列表格法do出来！
        // 先计算左下三角，再计算右上三角的乘积和！
        /*
        举例子画图说明：    1 2 3 4 5
        1  1  <== ① 2 3 4 5 ==> 120 ==> 1x120(t == 1x5x4x3x2) == 120  
        2  1  <== 1 ① 3 4 5 ==> 60  ==> 1x60(t == 1x5x4x3)  == 60
        3  2  <== 1 2 ① 4 5 ==> 20  ==> 2x20(t == 1x5x4)  == 40
        4  6  <== 1 2 3 ① 5 ==> 5   ==> 6x5(t == 1x5)   == 30
        5  24 <== 1 2 3 4 ① ==> 1   ==> 24x1(t == 1)  == 24
        */
        for(int i = 1;i < size;++i){// 不管index==0号的值！因为已经定了！
            res[i] = res[i-1] * a[i-1];
        }
        int t = 1;
        for(int i = size-2;i >= 0;--i){// 不管index==size-1号的值！因为已经定了！
            // 这里从后往前来求右上三级的乘积和太妙了吧！
            // 还有这下面的2行codes都非常棒！
            t *= a[i+1];
            res[i] *= t;
        }
        return res;
    }
};
```



#### <11> [LeetCode 剑指 Offer 57.和为s的两个数字](https://leetcode.cn/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

这道题目给我的启示：以后我自己刷题时一看到题目中==有已经排序过后数组==就要马上想起来用==双指针法！试一试看能不能deal题目！==

```c++
/*
    time:O(n)
    space:O(1)
*/
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        // 这是为看的评论区高赞大佬题解才知道能用《双指针》do出来的！
        // 这个题目意思原数组nums已经排序升序的数组了
        // 因此我就可以直接很容易就用 《双指针》 来deal的了！
        int l = 0,r = nums.size()-1;
        while(l < r){
            int tmp = nums[l]+nums[r];
            if(tmp < target)l++;// 太小了就l++让总和变大一点！
            else if(tmp > target)r--;// 太大了就r--让总和变小一点！
            else return {nums[l],nums[r]};// 刚好找到这个组合==target！
        }
        return {-1,-1};
    }
};
```



#### <12> [LeetCode 剑指 Offer 57 - II.和为s的连续正数序列](https://leetcode.cn/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

这个题目的==第一种暴力解法==是纯纯我自己想的回溯法do出来的！（暴力回溯搜索法！so虽然在LC上能够通过，但是时间和空间复杂度非常地不咋地！）


```c++
// 方法1：暴力回溯+判断谁是连续序列的剪枝方法的结合 题解 我个人认为还是不错的！（PS:因为是我自己一个人想出来的哈~）
class Solution {
public:
    vector<vector<int>> res;
    vector<int> tmp;
    void backtracking(int sum,int tar,int startIdx){
        if(sum == tar){
            if(tmp.size() > 1)res.push_back(tmp);
            return;
        }
        for(int i = startIdx;i < tar;++i){
            if(!tmp.empty() && tmp.back() != i - 1)break;
            // 不是连续序列子数组也剪枝！这行代码就保证了结果数组是连续的！
            if(sum + i > tar)break;// 剪枝！ 
            sum += i;
            // 因为[1~target]的数组一定是升序的了，当前加起来超过target了
            // 那么后面的加起来也一定是会超过的！放心吧！
            tmp.push_back(i);
            backtracking(sum,tar,i+1);
            sum -= i;// 回溯
            tmp.pop_back();
        }
    }
    vector<vector<int>> findContinuousSequence(int target) {
        // 这个题目应该是需要用到暴力搜索回溯法！
        res.clear();tmp.clear();
        backtracking(0,target,1);// startIdx == 1是因为0不算入累加的行列中!
        // 用回溯法弄出来all从[1~target)数组中和为target的字数组后
        //（注：就算是一个数字tar == tar了也不满足题意！因为题目说了是至少要有2个元素的和==tar！）
        // 同时在暴力回溯的同时判断谁是严格升序子数组序列即可了！
        return res;
    }
};
```



#### <13> [LeetCode 剑指 Offer 58 - I.翻转单词顺序](https://leetcode.cn/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

这个题目我之前2刷过carl哥的网站的题（lcT151），但是为忘记如何用双指针来do这个题目了！也就是说我虽然二刷了carl哥的除dp之外的题目，但是还是不能全覆盖地记住如何deal这些问题！

```c++
class Solution {
public:
    // 学carl哥之前教我的双指针法来do这套题！
    string reverseWords(string s) {
        // double pointer!
        int l = 0,size = s.size(),r = size - 1;
        // 1-先去除原字符串前后的多余空格子
        // 2-再一个一个解析word即可了！
        while(l < size && s[l] == ' ')l++;
        while(r >= 0 && s[r] == ' ')r--;
        string retStr = "";
        while(l <= r){
            int tmpIdx = r;// tmpIdx用以解析单个单词
            while(tmpIdx >= l && s[tmpIdx] != ' ')tmpIdx--;// 解析单个单词出来
            auto word = s.substr(tmpIdx+1,r - tmpIdx);
            retStr += word;
            // 将解析好的单个单词输入到结果字符串中（因为是从后往前遍历原字符串，因此可以借此构造出新的原字符串的反向字符串了！）
            if(tmpIdx > l)retStr.push_back(' ');// 补1个空格子
            while(tmpIdx >= l && s[tmpIdx] == ' ')tmpIdx--;
            // 去除中间各个单词之间的多余的空格子
            r = tmpIdx;// 更新右指针
        }
        return retStr;
    }
};
```



#### <14> [LeetCode 剑指 Offer 61 .扑克牌中的顺子](https://leetcode.cn/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

这个题目我是完全没有题解的思路，我自己的思路没法deal特殊case！不缺全面！很难想到这个题目的思路！真的！

```c++
// 这个顺子的题目总体的思路就是：
// 升序排序后比较最大最小牌值之差
// 小前提：构成顺子的这5个牌都必须是无重复元素的！
// if max - min < 5 ==> return true;
// else return false;

/*
   time:O(n),n是数组nums的大小
   space:O(1)
*/
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        int size = nums.size(),minIdx = 0;
        std::sort(nums.begin(),nums.end());
        for(int i = 0;i < size - 1;++i){
            if(nums[i] == 0)minIdx++;// 找到最小牌的index
            else if(nums[i] == nums[i+1])return false;// 非0重复就不构成顺子，提取返回false
        }
        return nums[size-1] - nums[minIdx] < 5;
        // 最大牌 - 最小牌 必须小于5才能构成顺子！
    }
};
```



#### <15> [LeetCode 剑指 Offer 64 .其1+2+...+n](https://leetcode.cn/problems/qiu-12n-lcof/)

这个题目我是完全没有想到不用if else while for switch case还有啥能do出来

然后看了题解才知道一种叫做==逻辑符号短路效应==的小知识点！

```c++
/*	本题用来著名的逻辑符号的短路效应：
	逻辑运算符的短路效应：
	if(A && B) 若 A 为 false ，则 B 的判断不会执行（即短路），直接判定 A && B 为 false
	if(A || B)若 A 为 true ，则 B 的判断不会执行（即短路），直接判定 A || B 为 true
	本题需要实现 “当 n=1时终止递归” 的需求，可通过短路效应实现。
 	n > 1 && sumNums(n - 1) // 当 n = 1 时 n > 1 不成立 ，此时 “短路” ，终止后续递归
*/
class Solution {
public:
    int res = 0;
    int sumNums(int n) {
        // 因为不能用乘除法，还有各种if else while 和for，因此都不能用了！
        // 只能用递归不停地调用自己来累加！
        bool t = (n > 1) && sumNums(n-1);// 若n==1甚至小于1了，就无需再开多一个递归函数了！
        res += n;
        return res;
    }
};
```



#### <16> [剑指 Offer 67. 把字符串转换成整数](https://leetcode.cn/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

这道题目其实我自己想的差不多了，就是差一点点判断越界的细节没想出来而已！

然后==参考了lc评论区的一个视频==，是B站up主https://www.bilibili.com/video/BV1mY411c7XJ/?vd_source=b050ab0adaffa51f5ad24d77efa40057的视频！然后才完全写出来的！

这道题目其实和[lc8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/)是一样的！复制黏贴过去也能AC！！！

```c++
class Solution {
public:
    unordered_map<char,int> hashMap;
    int strToInt(string str) {
        hashMap['0']++;hashMap['1']++;hashMap['2']++;hashMap['3']++;hashMap['4']++;
        hashMap['5']++;hashMap['6']++;hashMap['7']++;hashMap['8']++;hashMap['9']++;
        int size = str.size();
        if(size == 0)return 0;
        int idx = 0;// 用来do遍历的索引
        int PorN = 1;// 1表示正数 -1表示负数
        // 先去除字符串str前面多余的空格子！
        while(idx < size && str[idx] == ' ')idx++;
        // 判断特殊case + 符号判断
        if(idx < size){
            if(hashMap.find(str[idx]) == hashMap.end()){
                if(str[idx] == '-')PorN = -1;
                else if(str[idx] == '+')PorN = 1;
                else return 0;// 在数字前面若出现了非数字字符那这个题目默认就是返回0的
                idx++;
            }
        }
        long long res = 0;
        // 用long long作为res的类型是防止lc后台测试用例数字过大导致移除错误！
        for(;idx < size;++idx){
            int digit = str[idx] - '0';// 将每一个数字字符转换为对应的int数字(-'0'技巧)
            if(digit < 0 || digit > 9)break;// 遇到数字后面是非数字字符也马上break跳出计数循环
            res = res * 10 + digit;// 计数
            if(res * PorN >= INT_MAX)return INT_MAX;
            if(res * PorN <= INT_MIN)return INT_MIN;
        }
        return (int)res * PorN;
    }
};
```



#### <17> [剑指 Offer 42. 连续子数组的最大和](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

这个题目和[lc53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)是一毛一样的解题思路！我之前在carl哥那do过2次了，但是这次还是没有记住怎么do！

```c++
// time:O(n),n是原数组nums的大小！
// space:O(1)
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN,cnt = 0,size = nums.size();
        // res保存结果,cnt保存临时连续子数组和！
        for(int i = 0;i < size;++i){
            cnt += nums[i];
            res = res < cnt ? cnt:res;// 保存最大连续子数组和
            if(cnt < 0)cnt = 0;// 若cnt小于0了，就不符合连续子数组构成的最大和这个条件了
            // 就让cnt重新赋值为0，继续求连续和的最大值！
        }
        return res;
    }
};
```



#### <18> [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

本题我是看了评论区“递归和栈两种方式解决，最好的击败了100%”的用户这个标题的大佬才do出来的！

https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/solution/di-gui-he-zhan-liang-chong-fang-shi-jie-jue-zui-ha/

```c++
// 分治递归法：(左闭右闭区间)
// 思路：对于BST来说，后序遍历后最后一个元素必然是根节点root
// 然后后续遍历的第一个大于root的节点到root的前一个节点这个区间内的节点必然为BST的右子树
// 然后从前面到第一个大于root的节点的前一个节点的区间必然是BST的左子树
// 即：[left,第一个大于当前root节点-1]是BST的左子树区间
//    [第一个大于当前root节点,root节点-1]是BST的右子树区间
// 若不满足以上的2个条件的话，就一定不是BST了！
class Solution {
public:
    bool traversal(const vector<int>& p,int l,int r){
        if(l >= r)return true;
        int fenGeIdx = l;
        int curRoot = p[r];// 因为BST的后续遍历最后一个节点值肯定是当前根节点的值！so直接取p[r]==root
        while(p[fenGeIdx] < curRoot)fenGeIdx++;
        /*
            因为要确认当前的[第一个大于当前root节点,root节点-1]区间必须都要大于curRoot值
            才能确定是BST当前根节点curRoot的右子树！
            若有小于curRoot值的就马上return false表示不是BST了！
        */
        int tmp = fenGeIdx;
        while(tmp < r){// 右子树但凡有小于root的马上判断不是BST
            if(p[tmp++] < curRoot)return false;
        }
        /*
            同上理，要确定[left,第一个大于当前root节点-1]是BST的左子树区间这个区间的all节点值
            都要小于当前根节点curRoot的值！才符合BST！
        */
        tmp = fenGeIdx - 1;
        while(tmp >= l){// 左子树但凡有大于root的马上判断不是BST
            if(p[tmp--] > curRoot)return false;
        }
        // 判断左子树是否为BST
        bool leftIsBST = traversal(p,l,fenGeIdx-1);
        // 判断右子树是否为BST
        bool rightIsBST = traversal(p,fenGeIdx,r-1);
        // 进而判断整棵树是否为BST
        return leftIsBST && rightIsBST;
    }
    bool verifyPostorder(vector<int>& postorder) {
        return traversal(postorder,0,postorder.size()-1);
    }
};
```



#### <19> [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode.cn/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

##### [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

这两个题目是一毛一样的呀！我自己还是在carl哥的代码随想录的二叉树系列刷了很多次了的！！！还是没法完整写出来！

```c++
class Solution {
public:
    // 分析：因为我们此时要找出所有从根节点到叶子节点路径总和==给定目标和的路径
    // 因此我们必须要遍历整一颗二叉树，因此递归函数不需要返回值
    // 用递归函数从前到后用前序遍历的思路来do就符合找根节点到叶子节点的路径的思路了！
    // 这个思路来自carl哥的代码随想录！
    void traversal(TreeNode* cur,vector<int>& tpath,vector<vector<int>>& path,int tar){
        // 整体上是用到前序遍历 :中 左 右的logic来遍历二叉树的！
        if(!cur->left && !cur->right){
            // 遍历到根节点才算一个有效路径
            // 然后再看看是否符合路径和==目标和！
            if(tar == 0){
                path.push_back(tpath);
                return;// 符合目标和就加入结构路径path中，并继续遍历其他路径
            }else return;// 不符合，就继续回溯遍历其他路径
        }
        if(cur->left){
            // 处理当前节点
            tar -= cur->left->val;
            tpath.push_back(cur->left->val);
            // 继续递归
            traversal(cur->left,tpath,path,tar);
            // 回溯
            tar += cur->left->val;
            tpath.pop_back();
        }
        if(cur->right){
            // 处理当前节点
            tar -= cur->right->val;
            tpath.push_back(cur->right->val);
            // 继续递归
            traversal(cur->right,tpath,path,tar);
            // 回溯
            tar += cur->right->val;
            tpath.pop_back();
        }
    }
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if(root == nullptr)return {};
        vector<int> tpath;
        vector<vector<int>> path;
        tpath.push_back(root->val);// 不论如何你的整棵树的根节点root一定是先要加入到路径中的
        // 因为你题目说了，根节点到叶子节点的路径才算完整de有效路径！
        traversal(root,tpath,path,targetSum - root->val);
        return path;
    }
};
```



#### <20> [剑指 Offer 30. 包含min函数的栈](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof/)

##### [155. 最小栈](https://leetcode.cn/problems/min-stack/)

又是一个我在carl哥那刷过很多次的题目，我没有再次写出来！

```c++
class MinStack {
public:
    /** initialize your data structure here. */
    stack<pair<int,int>> stk;
    // stk存储的是一个队组，first表示元素值，second表示当前栈中存储的元素值中的最小值！
    MinStack() {
        
    }
    void push(int x) {
        if(stk.empty()){
            stk.push(pair<int,int>(x,x));
        }else{
            stk.push(pair<int,int>(x,std::min(x,stk.top().second)));
            // 因为与本结构中的成员函数min有重名，因此要加上std::调用标准的名字空间中的min来do区分！
        }
    }
    void pop() {
        if(stk.empty())return;
        stk.pop();
    }
    int top() {
        // 要是只用一个普通的stack<int>栈的话就不能正确弄出原本加入到top值！
        if(stk.empty())return -1;
        return stk.top().first;
    }
    int min() {
        if(stk.empty())return -1;
        return stk.top().second; 
    }
};
```



#### <21> [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode.cn/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

##### [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

这又是一道我在carl哥那刷过的题目，但是我忘记怎么做了！

```c++
class Solution {
public:
    /* 
    思路：求BT的最近公共祖先，就必须要想到要用回溯法！也即后续遍历BT的logic
        从下往上遍历！当 当前节点是空or p or q时直接返回cur
        然后遍历其左右子树
        当左右都不空，说明当前节点就是最近公共祖先，直接返回即可
        当左空右不空，说明最近公共祖先节点会由右子树直接返回
        当左不空右空，说明最近公共祖先节点会由左子树直接返回
        当左右都为空，直接返回左or右，此时没有在以当前cur节点为根节点的树的左右子树中找到公共祖先节点
        so直接返回nullptr
    */
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr || root == q || root == p)return root;
        TreeNode* left = lowestCommonAncestor(root->left,p,q);
        TreeNode* right = lowestCommonAncestor(root->right,p,q);
        if(left && right)return root;
        else if(left && !right)return left;
        else if(!left && right)return right;
        return right; 
    }
};
```



#### <22> [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

##### [235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

这还是一道我在carl哥那刷过的题目，但是我忘记怎么做了！

```c++
// way1：迭代法求BST的最近公共祖先节点！
// 这里利用了BST的节点的值的有序性！左->val < 根->val < 右->val <==> 根是左和右的最近公共祖先节点了！
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr)return root;
        while(root){
            if(root->val < p->val && root->val < q->val){
                root = root->right;// 去右边找最近公共祖先节点！
            }else if(root->val > p->val && root->val > q->val){
                root = root->left;// 去左边找最近公共祖先节点！
            }else return root;// 当前节点就是p和q的最近公共祖先节点了！
        }
        return nullptr;// 迭代找不到最近公共祖先节点的话就直接返回空nullptr即可！
    }
};
// way2：当然你也可以把BST当作是普通二叉树那样，用递归来求最近公共祖先节点！
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr || root == q || root == p)return root;// 空节点的最近公共祖先肯定也是空节点！    
        TreeNode* left = lowestCommonAncestor(root->left,p,q);
        TreeNode* right = lowestCommonAncestor(root->right,p,q);
        if(left && right)return root;
        else if(left && !right)return left;
        else if(!left && right)return right;
        return left;
    }
};
```



#### <23> [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

*学的这个lc题目评论区的高赞答案写的！！！太NB了！可太难想了这个思路！*

==思路：==

1- 先在树A中找到头结点 == 树B头结点的节点

2- 找到之后再将此时的A子结构的left和right 与 B的left和right逐个do队比，即可判断A是否含有B这样的子结构了！

```c++
class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(A == nullptr || B == nullptr)return false;// 空子树不是任何一棵树的子结构
        if(A->val == B->val){// 只有找到同一个根节点时才能接下去判断是否匹配的问题！
            if(traversal(A,B))return true;
        }
        // A的左orright 只有含有B这样的结构就都算是true
        return isSubStructure(A->left,B) || isSubStructure(A->right,B);
    }
    // 下面这个子函数是判断这2个可能的子结构A和B到底是否是匹配的
    bool traversal(TreeNode* A,TreeNode* B){
        if(B == nullptr)return true;// 若判断出B遍历完成了，那肯定子结构A与B相互匹配了！
        if(A == nullptr || A->val != B->val)return false;
        // 若判断出A遍历完成了(到A的叶子节点了，再往该节点的左右遍历都是空了)
        // or 说遍历到当前A的子结构该节点的值！=B对应节点的值，那肯定子结构A与B相互不匹配了！
        // 此时就是A子结构的当前节点和B的对应节点值相等的case，就继续递归遍历其他节点的值
        // 看是否匹配！
        return traversal(A->left,B->left) && traversal(A->right,B->right);
    }
};
```



#### <24> [剑指 Offer 16. 数值的整数次方](https://leetcode.cn/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

##### [50. Pow(x, n)](https://leetcode.cn/problems/powx-n/)这个lc50也同上！是同一个题目！

本题的方法被称为==「快速幂算法」==!我学自lc官方题解区的官方视频题解！！！忘了就回看一下官方给出的视频讲解即可！https://leetcode.cn/problems/powx-n/solution/powx-n-by-leetcode-solution/

==快速幂次算法==本质就是==分治算法==，时间复杂度就是O（logn）的！

思路：

分治算法：（通过下面这2个例子详细你自己也能够悟出来的了！）

x ^16 <=> x^16 -> x^8 * x^8 -> x^4 * x^4 * x^4 * x^4  ->  x^2 * x^2 * x^2 * x^2 * x^2 * x^2 * x^2 * x^2

x ^20 <=> x^20 -> x^10 * x^10-> x^5 * x^5 * x^5 * x^5 ->  x^2 * x^2 * x *x^2 * x^2 * x

然后思路就是递归分治！（若是理解了上面这2个例子要写出递归分治的codes不难了！）

当n < 0时，取result == 1.0 / pow(x,abs(n));

当n >= 0时，取result == pow(x,n);

```c++
// 时间：O(logn)，递归的层数
// 空间：O(logn)，递归的层数，这是由于递归的函数调用会使用栈空间。
class Solution {
public:
    double myPowHelper(double x, uint32_t n) {
        // uint32_t来表示n才能不越界！NB！！！当然你甚至可以用uint64_t也ok！
        if(n == 0)return 1.0;// 当n递减到0时就代表x到最后了！就直接x个1.0完事了！
        if(n % 2 != 0){
            double half = myPowHelper(x,n / 2);
            return half * half * x;
        }else{
            double half = myPowHelper(x,n / 2);
            return half * half;
        }
    }
    double myPow(double x,int n){
        // 必须要考虑到n == 0 和 x == 1时这种特殊case！
        if(n == 0 || x == 1)return 1;
        // 任何数的0次方 or 1的任何次方都 == 1！
        if(n < 0){
            return 1.0 / myPowHelper(x,abs(n));
        }
        return myPowHelper(x,n);
    }
};
```

#### <25> [剑指 Offer 07. 重建二叉树](https://leetcode.cn/problems/zhong-jian-er-cha-shu-lcof/)

这个题目我在carl哥那也刷了N遍了，但是还是不会do！那就背下来把！！！

```c++
 /*
    思路：先（每次递归都用前序的第一个元素来）切割中序，后（每次递归都用中序左的size来）切割前序列
    		[3 9 20 15 7]
            [9 3 15 20 7]
            /           \ 
           [9]          [20 15 7]    
           [9]          [15 20 7]
                       /        \
                     [15]     [7]
                     [15]     [7]
 */
class Solution {
public:
    TreeNode* traversal(vector<int>& preorder, vector<int>& inorder) {
        int size = preorder.size();
        if(size == 0)return nullptr;
        TreeNode* root = new TreeNode(preorder[0]);
        if(size == 1)return root;
        // 先切割中序(按照左开右闭原则)
        int qiegeIdx = -1;
        for(int i = 0;i < inorder.size();++i){
            if(inorder[i] == preorder[0]){
                qiegeIdx = i;break;
            }
        }
        vector<int> leftInorder(inorder.begin(),inorder.begin()+qiegeIdx);
        vector<int> rightInorder(inorder.begin()+qiegeIdx+1,inorder.end());
        // 后切割前序(按照左开右闭原则)
        preorder.erase(preorder.begin(),preorder.begin()+1);// 先物理删除用过的前序第一个元素值（当前根节点值）
        int leftInorderSize = leftInorder.size();
        vector<int> leftPreorder(preorder.begin(),preorder.begin()+leftInorderSize);
        vector<int> rightPreorder(preorder.begin()+leftInorderSize,preorder.end());
        // 用递归函数返回值do重新构造二叉树的工作！
        root->left = traversal(leftPreorder,leftInorder);// 继续构造左子树
        root->right = traversal(rightPreorder,rightInorder);// 继续构造右子树
        return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.size() == 0 || inorder.size() == 0)return nullptr;
        return traversal(preorder,inorder); 
    }
};
```



下面这个题目与上面的思路类似：

##### [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```c++
 /*
    思路：先（每次递归都用后序的last一个元素来）切割中序，后（每次递归都用中序左的size来）切割后序列
            [9 3 15 20 7]
            [9 15 7 20 3]
            /           \ 
           [9]          [15 20 7]    
           [9]          [15 7 20]
                       /        \
                     [15]      [7]
                     [15]      [7]
 */
class Solution {
public:
    TreeNode* traversal(vector<int>& inorder, vector<int>& postorder){
        int size = postorder.size();
        if(size == 0)return nullptr;
        TreeNode* root = new TreeNode(postorder[size-1]);
        if(size == 1)return root;
        // 若当前序列数组只有一个元素，就直接返回把！不用继续递归构建BT了！
        // 先切割中序
        // 1:找切割中序的切割点
        int qiegeIdx = -1;
        for(int i = 0;i < inorder.size();++i){
            if(inorder[i] == root->val){
                qiegeIdx = i;break;
            }
        }
        // 2:切割中序(以左闭右开原则)
        vector<int> leftInorder(inorder.begin(),inorder.begin()+qiegeIdx);
        vector<int> rightInorder(inorder.begin()+qiegeIdx+1,inorder.end());
        // 后切割后序(以左闭右开原则)
        postorder.resize(postorder.size()-1);// 物理上pop掉最后一个用过的元素值（是当前根节点）
        vector<int> leftPostorder(postorder.begin(),postorder.begin()+leftInorder.size());
        vector<int> rightPostorder(postorder.begin()+leftInorder.size(),postorder.end());
        // 利用递归函数返回值do重新构造二叉树的工作
        root->left = traversal(leftInorder,leftPostorder);
        root->right = traversal(rightInorder,rightPostorder);
        return root; 
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        if(inorder.size() == 0 || postorder.size() == 0)return nullptr;
        return traversal(inorder,postorder);
    }
};
```



#### <26> [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode.cn/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

这个困难题目就当让我长长见识得了！太难了！说实话面试要是遇到了那也只能G了！

```c++
// 这个题目我是看的b站up主香辣鸡排蛋包饭才知道这么do的！
// 但是这也太难了吧,我说实话，只能把这种思路完全记忆背下来才能默写出来！
/*
    time:O(1),就算是百万位的数字，也只有需要遍历7次，因为只有7位的数字！O(7) == O(1)
    space:O(1)
*/
class Solution {
   public:
    int countDigitOne(int n) {
        int res = 0;// 保存结果
        long base = 1;// 从个位开始统计
        // 对于数字n，要 求从1~n的all数字中出现'1'的次数 == 求sum(各个位数上面出现的'1'的次数)
        while(base <= n){
            int a = n / base / 10;
            int b = n % base;
            int cur = n / base % 10;
            if(cur > 1){
                res += (a+1)*base;
            }else if(cur == 1)res += a*base + b + 1;
            else res += a*base;
            base *= 10;// 依次统计各个位置上到‘1’出现的次数！
        }
        return res;
    }
};
```



#### <27> [剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

这道题目虽然是困难级别的，但还是能做的，不像上面<26>题那样变态！！！虽说也挺难想的，但是我做这个题目的同时也复习了一下归并排序如何写！so这个题目还是值得一做的！

==这道题目主要参考的是B站up主：香辣鸡排蛋包饭的video讲解！==

https://www.bilibili.com/video/BV1CK411c7gx?p=45&vd_source=b050ab0adaffa51f5ad24d77efa40057

还有一篇lc题解区的小题解：

https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/solution/jian-zhi-offer-51-shu-zu-zhong-de-ni-xu-a80cq/

```c++
/*
    考察的知识点：归并排序模板默写的同时，统计逆序对！
    归并排序会将数组各个元素值do升序排序（默认），同时我们再统计一下逆序对即可over 本题了！
    当有 ：if(nums[l_pos] > nums[r_pos])时，马上让 res += (mid - l_pos + 1);即可统计逆序对了！
*/
class Solution {
private:
    int res = 0;
public:
    int reversePairs(vector<int>& nums) {
        //使用归并排序法来do这道题目！
        vector<int> tmp;
        int size = nums.size();
        tmp.resize(size);// 必须要提前resize，否则lc就是会给你判错！
        if(size == 0 || size == 1)return 0;// 特殊case，此时没有逆序对
        mergeSort(nums,0,size-1,tmp);
        return res;
    }
    void mergeSort(vector<int>& nums,int left,int right,vector<int>& tmp){
        if(left >= right)return;// 一个元素本身就是有序的了！
        int mid = left + (right - left) / 2;
        // 递归分割左区间
        mergeSort(nums,left,mid,tmp);
        // 递归分割右区间
        mergeSort(nums,mid+1,right,tmp);
        // 合并,同时统计逆序对的个数res
        merge(nums,left,mid,right,tmp);
    }
    void merge(vector<int>& nums,int left,int mid,int right,vector<int>& tmp){
        int l_pos = left,r_pos = mid+1,pos = left;
        while(l_pos <= mid && r_pos <= right){
            if(nums[l_pos] <= nums[r_pos])tmp[pos++] = nums[l_pos++];
            else {
                res += (mid - l_pos+1);// 在Cpp版本的归并排序模板中就写这一句代码就能通过此题！
                tmp[pos++] = nums[r_pos++];
            }
        }
        while(l_pos <= mid)tmp[pos++] = nums[l_pos++];
        while(r_pos <= right)tmp[pos++] = nums[r_pos++];
        for(int i = left;i <= right;++i)nums[i] = tmp[i];
    }
};
```

补充：==归并排序模板==

归并排序B站大佬讲解的video：https://www.bilibili.com/video/BV1Pt4y197VZ?spm_id_from=333.337.search-card.all.click&vd_source=b050ab0adaffa51f5ad24d77efa40057

```c++
// 1-Cpp版归并排序模板：
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<algorithm>
using namespace std;
void merge(vector<int>& nums,int left,int mid,int right,vector<int>& tmp){
    int l_pos = left,r_pos = mid+1,pos = left;
    while(l_pos <= mid && r_pos <= right){
        if(nums[l_pos] <= nums[r_pos]){
            tmp[pos++] = nums[l_pos++];
        }
        else {
            tmp[pos++] = nums[r_pos++];
        }
    }
    while(l_pos <= mid)tmp[pos++] = nums[l_pos++];
    while(r_pos <= right)tmp[pos++] = nums[r_pos++];
    for(int i = left;i <= right;++i)nums[i] = tmp[i];
}
void mergeSort(vector<int>& nums,int left,int right,vector<int>& tmp){
    if(left >= right)return;// 一个元素本身就是有序的了！
    int mid = left + (right - left) / 2;
    // 递归分割左区间
    mergeSort(nums,left,mid,tmp);
    // 递归分割右区间
    mergeSort(nums,mid+1,right,tmp);
    // 合并
    merge(nums,left,mid,right,tmp);
}
void print(const vector<int>& nums){
    for_each(nums.begin(),nums.end(),[](int v){cout<<v<<"\t";});
    cout<<endl;
}
int main(int argc,char* argv[]){
    vector<int> nums{9,5,2,7,12,4,3,1,11};// {7,5,6,4};
    vector<int> tmp;
    tmp.resize(nums.size());
    print(nums);
    mergeSort(nums,0,nums.size()-1,tmp);
    print(nums);
    return 0;
}
/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
// 2-C语言版归并排序模板：
#include<stdio.h>
#include<stdlib.h>
void print(int * arr,int n){
    for(int i = 0;i < n;++i){
        printf("%d\t",arr[i]);
    }
    printf("\n");
}
// 合并函数
void merge(int* arr,int* tmpArr,int left,int mid,int right)
{
    // 标记左半区第一个未排序的元素de index
    int l_pos = left;
    // 标记右半区第一个未排序的元素de index
    int r_pos = mid + 1;
    // 临时数组元素的下标
    int pos = left;
    // 合并
    while(l_pos <= mid && r_pos <= right){
        if(arr[l_pos] < arr[r_pos]){
            tmpArr[pos++] = arr[l_pos++];
        }else{
            tmpArr[pos++] = arr[r_pos++];
        }
    }
    // 合并左半区剩余的元素
    while(l_pos <= mid){
        tmpArr[pos++] = arr[l_pos++];
    }
    // or合并右半区剩余的元素
    while(r_pos <= right){
        tmpArr[pos++] = arr[r_pos++];
    }
    // 把临时数组中合并后的元素复制回原来的数组
    for(int i = 0;i <= right;++i){
        arr[i] = tmpArr[i];
    }
}
// 归并排序的实质函数
void msort(int * arr,int * tmpArr,int left,int right){
    // 如果只有一个元素的区域就不需要继续划分了
    // 只需要被归并，因为一个元素的区域本身就是有序的了！无需排序了！
    if(left >= right)return;
    // 找中间点
    int mid = left + (right - left) / 2;
    // 递归划分左半区域
    msort(arr,tmpArr,left,mid);
    // 递归划分右半区域
    msort(arr,tmpArr,mid+1,right);
    // 合并已经排好序的部分区域
    merge(arr,tmpArr,left,mid,right);
}
// 归并排序的入口函数
void merge_sort(int * arr,int n){
    int* tmpArr = (int*)malloc(n * sizeof(int));
    if(tmpArr){
        msort(arr,tmpArr,0,n-1);
        free(tmpArr);
    }else{
        printf("error,failed to allocate tmp memory!\n");
    }
}
int main(int argc,char* argv[]){
    int arr[] {9,5,2,7,12,4,3,1,11};
    int n = 9;
    print(arr,n);
    // 归并算法总体思路：先递归切分然后再排序合并即可！
    merge_sort(arr,n);
    print(arr,n);
    return 0;
}
```

#### <28> [剑指 Offer 45. 把数组排成最小的数](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

这个题目的题解是非常巧妙的！！！利用了==字符串相加后再比大小==的技巧！还是从lc题解区的路飞k神大佬那学的！！！这个技巧还是能长长见识，然后记忆下来的！

<img src="F:\学习过程中的截图\lc剑指offer45.png" alt="lc剑指offer45" style="zoom: 67%;" />

```c++
struct myPred:public binary_function<string,string,bool>{
    bool operator()(const string& x,const string& y)const{
        return (x + y) < (y + x);// 拼接成输出最小的数字字符串的自定义排序谓词！
    }
};
class Solution {
public:
    string minNumber(vector<int>& nums) {
        vector<string> strs;
        for(int& num : nums){// 先将每个数字转换为字符串，并加入到字符串数组中去！
            strs.push_back(to_string(num));
        }
        // 按照// 自定义排序拼接成输出最小的数字！
        std::sort(strs.begin(),strs.end(),myPred());
        string res = "";
        for(auto str:strs){// 弄成一个字符串好do return
            res += str;
        }
        return res;
    }
};
```



#### <29> [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode.cn/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

这个题目和[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)是一样的！

```c++
// 滑动窗口（双指针法）,这是学的y总的！
// 得自己按照lc给出的特例，比如第一个特例，拿来跟着我这份代码画个图就行了！
/*
	time：O(n)
	space:O(n)
*/
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int res = 0,size = s.size();
        unordered_set<char> unset;
        for(int i = 0,j = 0;i < size;++i){
            // 只要发现是重复元素就继续删除掉set中的对应元素
            while(unset.find(s[i]) != unset.end()){
                unset.erase(s[j++]);
            }
            unset.insert(s[i]);// 继续加入新元素
            res = max(res,i-j+1);// 更新最长不重复数组长度！
        }
        return res;
    }
};
```





## **牛客网  《面试必刷TOP101 》刷题笔记**

这里的题目都是我自己一刷时没do出来的（看题解） or do的有些磕磕绊绊需要想挺久才do出来的题目！！！（暗示我自己这些剑指offer的题目为肯定是需要2刷甚至3or4刷的！）

我到时候面试的时候就一定要先把==10大排序算法==再背回来！清除知道了解他们的时间空间复杂度！==(這一點是我在刷lc题目的时候没注意到的！)==



### 目前累计总共有==《5》==道题：



#### <1> [BM2-链表内指定区间反转](https://www.nowcoder.com/practice/b58434e200a648c589ca2063f1faf58c?tpId=295&tqId=654&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

<img src="C:/Users/11602/Desktop/LeetCodeOfferNotes/git/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E6%88%AA%E5%9B%BE/1.jpg" alt="1" style="zoom: 50%;" />

```c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 * };
 */
// time:O(n),最坏情况下，需要遍历整个list，比如m or n 是指向最后一个节点时
// spcae:O(1),占用了常量级别的指针内存空间
class Solution {
public:
    /**
     * 
     * @param head ListNode类 
     * @param m int整型 
     * @param n int整型 
     * @return ListNode类
     */
    // 我说实话，这个题目我看了官方题解之后真的觉得很nb的！（画个图就出来了！）
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        ListNode* dummyNode = new ListNode(-1);// 虚拟头结点
        dummyNode->next = head;
        ListNode* prev = dummyNode;
        ListNode* cur = head;
        // 找到m这个位置的节点
        for(int i = 1;i < m;++i){
            prev = cur;
            cur = cur->next;
        }
        // 从m反转到n
        for(int i = m;i < n;++i){
            ListNode* tmp = cur->next;
            cur->next = tmp->next;
            tmp->next = prev->next;
            prev->next = tmp;
        }
        return dummyNode->next;// 返回真正的list头结点！
    }
};
```



#### <2> [BM3-链表中的节点每k个一组翻转](https://www.nowcoder.com/practice/b49c3dc907814e9bbfa8437c251b028e?tpId=295&tqId=722&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj)

```c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 * };
 */
// time:O(n),最坏情况下需要遍历整个list，space:O(1),只占用了常量指针内存空间！
class Solution {
public:
    /**
     * 
     * @param head ListNode类 
     * @param k int整型 
     * @return ListNode类
     */
    // 思路：每k个节点do一次部分反转list的操作即可！
    // 		运用到了属于BM2中部分反转list的操作！(个人认为我自己基于BM2这个题解写的思路比BM3要好！)
    ListNode* reverseKGroup(ListNode* head, int k) {
        int size = 0;
        ListNode* cur = head;
        // 求list size
        while(cur){
            cur = cur->next;
            size++;
        }
        // 处理下特殊case：size == k 时应该让list全反转
        if(size == k){
            head = reverseAll(head);
            return head;
        }
        // size != k 时应该让list部分反转
        int i = 1;
        while(1){
            if( (i+k-1) >= size)break;
            head = subReverseFun(head,i,i+k-1);
            i += k;
        }
        return head;
    }
    ListNode* subReverseFun(ListNode* head,int m,int n){
        ListNode* dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode* prev = dummyNode;
        ListNode* cur = head;
        // 找到m位置的节点
        for(int i = 1;i < m;++i){
            prev = cur;
            cur = cur->next;
        }
        // 从m反转到n
        for(int i=m;i<n;++i){
            ListNode* tmp = cur->next;
            cur->next = tmp->next;
            tmp->next = prev->next;
            prev->next = tmp;
        }
        return dummyNode->next;
    }
    // list全反转
    ListNode* reverseAll(ListNode* head){
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while(cur){
            ListNode* tmp = cur->next;
            cur->next = prev;
            prev = cur;
            cur = tmp;
        }
        return prev;
    }
};
```



#### <3> [BM7-链表中环的入口结点](https://www.nowcoder.com/practice/253d2c59ec3e4bc68da16833f79a38e4?tpId=295&tqId=23449&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目的哈希表法是我之前没想到的，但是双指针法（快慢指针法）我是了然如胸了！

```c++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
// way1:双指针法（快慢指针法，很好想的，你其实画个图试几下就出来的了）
// time:O(n),最坏情况下是无环遍历了整个list，space:O（1）,常数空间
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead) {
        // 分2步走：
        // 1-先判断是否有环，无环就马上返回nullptr，有环才继续进行步骤2
        // 2-找环的入口节点指针并返回（双指针法）
        
        // procedure1:
        ListNode* cur1 = pHead,* cur2 = pHead;
        bool hasCircuit = false;// 判断有无环的标志
        while(cur2->next && cur2->next->next){
            cur1 = cur1->next;
            cur2 = cur2->next->next;
            if(cur1 == cur2){
                hasCircuit = true;break;
            }
        }
        if(hasCircuit==false)return nullptr;
        // procedure2:
        ListNode* cur3 = pHead,* cur4 = cur2;
        while(cur3 != cur4){
            cur3 = cur3->next;
            cur4 = cur4->next;
        }
        return cur3;// 此时跳出循环后必然能够找到环入口节点指针！
    }
};

// way2:哈希表法(甚至比双指针法还好想的呢！)
// time:O(n),最坏情况下是无环遍历了整个list，space:O（n）,最坏情况下list中的all节点都存入了uset中
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead) {
        // 思路：将list中all的节点都加入到哈希表中，若当前节点存在于哈希表中则表明存在环，
        // 并且这个第一个存在于哈希表中的节点就是环的入口节点了！
        unordered_set<ListNode*> uset;
        ListNode* cur = pHead;
        while(cur){// 遍历list的同时将每个节点加入到哈希表中
            auto it = uset.find(cur);
            if(it != uset.end())return (*it);
            uset.insert(cur);
            cur = cur->next;
        }
        return nullptr;// 无环，返回空指针节点
    }
};
```



#### <4> [BM11-链表相加(二)](https://www.nowcoder.com/practice/c56f6c70fb3f4849bc56e33ff2a50b6b?tpId=295&tqId=1008772&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

这个题目我第一次do的时候确实没do对！（有暴力的思路也写出来代码了但是通过不了！）

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
// time:O(max(len1,len2))+O(m)+O(n) == O(max(m,n)),反转操作会遍历list1和list2各一次,so
// 取最长的长度即为新的结果list的长度了！
// space:O(1),常量级指针，没有额外的辅助空间
class Solution {
public:
    /**
     * 
     * @param head1 ListNode类 
     * @param head2 ListNode类 
     * @return ListNode类
     */
    ListNode* addInList(ListNode* head1, ListNode* head2) {
        // 看了官方的answer，感觉贼牛逼！
        // 思路：使用了反转list + 加法进位
        
        // 先处理特殊case：
        if(head1==nullptr)return head2;// 一个list为空，相当于0+任何数==任何数！
        if(head2==nullptr)return head1;// 同理
        // 后反转2个list
        head1 = listReverse(head1);
        head2 = listReverse(head2);
        int jinwei = 0;// 进位
        ListNode* newListHead = new ListNode(-1);// 结果list的头节点
        ListNode* cur = newListHead;// 必须要有一个cur节点来遍历！留着这个虚拟头结点do最后的反转操作！
        while(head1 != nullptr || head2 != nullptr){// 2个list中只要有一个不为空就继续do list的加法！
            int val = jinwei;// val记录了每次2个list对应node处->val之和！
            if(head1){
                val += head1->val;
                head1 = head1->next;
            }
            if(head2){
                val += head2->val;
                head2 = head2->next;
            }
            jinwei = val / 10;// 更新进位 的值
            int newVal = val % 10;// push进新的结果list的值！
            cur->next = new ListNode(newVal);
            cur = cur->next;
        }
        if(jinwei != 0){// 只要进位不为0，说明还是有进位值，需要多给list加个新的进位节点，且 节点值为1！
            cur->next = new ListNode(jinwei);
            cur = cur->next;// cur走不走多一步其实都无所谓了的，你画个图即可理解了！
        }
        return  listReverse(newListHead->next);;
    }
    ListNode*  listReverse(ListNode* head){// 子函数，用于反转list，便于从后往前do list->val 的加法！
        if(head==nullptr)return head;
        ListNode* cur = head;
        ListNode* prev = nullptr;
        while(cur){
            ListNode* tmp = cur->next;
            cur->next = prev;
            prev = cur;
            cur = tmp;
        }
        return prev;
    }
};
```



#### <5> [BM13-判断一个链表是否为回文结构](https://www.nowcoder.com/practice/3fed228444e740c8be66232ce8b87c2f?tpId=295&tqId=1008769&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)

傻了我，一时间居然没do出来这种简单的题目！

```c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 * };
 */
// 这个方法超时了！
// class Solution {
// public:
//     /**
//      * 
//      * @param head ListNode类 the head
//      * @return bool布尔型
//      */
//     bool isPail(ListNode* head) {
//         // write code here
//         ListNode* tmp = reverseList(head);
//         ListNode* left = head,* right = tmp;
//         while(left && right){
//             if(left->val != right->val)return false;
//             left = left->next;
//             right = right->next;
//         }
//         return true;
//     }
//     ListNode* reverseList(ListNode* head){
//         ListNode* cur = head,* prev = nullptr;
//         while(cur){
//             ListNode* tmp = cur;
//             cur->next = prev;
//             prev = cur;
//             cur = tmp;
//         }
//         return prev;
//     }
// };

// time:O(n),n是list的长度；space:O(n),n是list的长度；
class Solution {
public:
    bool isPail(ListNode* head) {
        // 思路：将list的元素们转换为数组，用数组是否是回文来判断list是否是回文即可！
        vector<int> v;
        while(head){
            v.push_back(head->val);
            head = head->next;
        }
        return isHuiwen(v);
    }
    bool isHuiwen(const vector<int>& v){
        int l = 0,r = v.size()-1;
        while(l < r){
            if(v[l] != v[r])return false;
            l++,r--;
        }
        return true;
    }
};
```





#### <6> [BM13-判断一个链表是否为回文结构](https://www.nowcoder.com/practice/3fed228444e740c8be66232ce8b87c2f?tpId=295&tqId=1008769&ru=/exam/oj&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D295)
