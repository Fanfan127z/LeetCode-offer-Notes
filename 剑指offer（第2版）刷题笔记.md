## **剑指offer（第2版）刷题笔记**

这里的题目都是我自己一刷时没do出来的（看题解） or do的有些磕磕绊绊需要想挺久才do出来的题目！！！（暗示我自己这些剑指offer的题目为肯定是需要2刷甚至3or4刷的！）

我到时候面试的时候就一定要先把==10大排序算法==再背回来！清除知道了解他们的时间空间复杂度！==(這一點是我在刷lc题目的时候没注意到的！)==

目前累计总共有==《15》==道题：

- [x] #### [LeetCode 剑指offer 04.二维数组中的查找：](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

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

- [x] #### [LeetCode 剑指offer 25.合并两个排序的链表：](https://leetcode.cn/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

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

- [x] #### [LeetCode 剑指offer 31.栈的压入、弹出序列：](https://leetcode.cn/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

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



- [x] #### [LeetCode 剑指offer 50.第一个只出现一次的字符：](https://leetcode.cn/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

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



- [x] #### [LeetCode 剑指offer 35.复杂链表的复制：](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

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

- [x] #### [LeetCode 剑指offer 54.二叉搜索树的第k大节点：](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

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



- [x] #### [LeetCode 剑指offer 55II.平衡二叉树：](https://leetcode.cn/problems/ping-heng-er-cha-shu-lcof/)

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



- [x] #### [LeetCode 剑指offer 56I.数组中数字出现的次数](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

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

#### [LeetCode 剑指 Offer II 031. 最近最少使用缓存](https://leetcode.cn/problems/OrIXps/)

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
		put(it->first, res);			// 头插查回去LRU中
		return res;
	}
	// put方法：将最近使用过的或者新的pair<key,val>插入到LRU的头部！
	void put(int key, int val) {
		// 找到
		// 找不到
		Node* tnode = new Node(key, val);
		auto it = map.find(key);
		if (it != map.end()) {// 找到有重复key时
			dlist->remove(it->second);	// 先删除双向list中的该节点（）
			dlist->addFirst(tnode);		// 后再向双向list的头部添加该节点
			map[key] = tnode;			// 再重新do哈希表的映射关系
		}
		else {
			if (map.size() == this->_capacity) {
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



- [x] #### [LeetCode 剑指 Offer 66.构建乘积数组](https://leetcode.cn/problems/gou-jian-cheng-ji-shu-zu-lcof/)

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



- [x] #### [LeetCode 剑指 Offer 57.和为s的两个数字](https://leetcode.cn/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

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



- [x] #### [LeetCode 剑指 Offer 57 - II.和为s的连续正数序列](https://leetcode.cn/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

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



- [x] #### [LeetCode 剑指 Offer 58 - I.翻转单词顺序](https://leetcode.cn/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

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



- [x] #### [LeetCode 剑指 Offer 61 .扑克牌中的顺子](https://leetcode.cn/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

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



- [x] #### [LeetCode 剑指 Offer 64 .其1+2+...+n](https://leetcode.cn/problems/qiu-12n-lcof/)

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



- [x] #### [剑指 Offer 67. 把字符串转换成整数](https://leetcode.cn/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

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



- [x] #### [剑指 Offer 42. 连续子数组的最大和](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

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

