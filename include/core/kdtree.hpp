#ifndef KDTREE_HPP_
#define KDTREE_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace epf {

enum class Iteration {
  InOrder,
};

// temporary: move outside of kdtree
template <typename Tree>
class KDTreeIterator {
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type        = typename Tree::Node;
  using pointer           = typename Tree::Node*;
  using reference         = typename Tree::Node&;
  using difference_type   = std::ptrdiff_t;

  friend Tree;

  Tree const* tree_;
  value_type const* current_;

  explicit KDTreeIterator(Tree const* t_tree, value_type const* t_current = nullptr)
    : tree_(t_tree), current_(t_current) {}

 public:
  [[nodiscard]] bool operator==(KDTreeIterator const& t_rhs) const { return this->current_ == t_rhs.current_; }

  [[nodiscard]] bool operator!=(KDTreeIterator const& t_rhs) const { return this->current_ != t_rhs.current_; }

  [[nodiscard]] value_type const& operator*() const { return *this->current_; }

  KDTreeIterator& operator++() {
    if (this->current_->child_node_[1] != nullptr) {
      this->current_ = this->current_->child_node_[1].get();
      while (this->current_->child_node_[0] != nullptr) {
        this->current_ = this->current_->child_node_[0].get();
      }
    } else {
      auto parent = this->current_->parent_;
      while (parent != nullptr and this->current_ == parent->child_node_[1].get()) {
        this->current_ = parent;
        parent         = parent->parent_;
      }

      this->current_ = parent;
    }

    return *this;
  }

  KDTreeIterator& operator--() {
    if (this->current_->child_node_[0] != nullptr) {
      this->current_ = this->current_->child_node_[0].get();
    } else {
      auto parent = this->current_->parent_;
      while (parent != nullptr and this->current_ != parent->child_node_[1].get()) {
        this->current_ = parent;
        parent         = parent->parent_;
      }

      this->current_ = parent;
    }

    return *this;
  }
};

template <typename T, typename ValueComp>
class KDTree {
  using ValueType   = T;
  using ElementType = double;       // usually double, if encounter other cases, inject to T
  using IndexType   = std::size_t;  // usually std::size_t, same as above

  friend class KDTreeIterator<KDTree>;

 public:
  class Node {
    using PivotType = std::pair<ElementType, IndexType>;

    friend class KDTree;
    friend class KDTreeIterator<KDTree>;

    Node* parent_ = nullptr;
    std::array<std::unique_ptr<Node>, 2> child_node_;

    ValueType pt_;
    ValueComp pt_comp_;

    PivotType pivot_   = {0, std::numeric_limits<IndexType>::max()};
    std::size_t depth_ = 0;

    Node* insert(KDTree* const t_base, ValueType const& t_pt) noexcept {
      // ros-navigation check if kdtree node is leaf, since they copy parent to child when inserting new node, and only
      // update the leaf node
      if (std::equal_to<ValueType>{}(this->pt_, t_pt)) {
        return nullptr;
      }

      auto& node_to_operate = pt_comp_(this->pt_, t_pt, &this->pivot_) ? this->child_node_[0] : this->child_node_[1];
      if (node_to_operate != nullptr) {
        return node_to_operate->insert(t_base, t_pt);
      }

      t_base->leaf_count_ += static_cast<std::size_t>(not this->is_leaf());
      node_to_operate          = std::make_unique<Node>(t_pt, this->pt_comp_);
      node_to_operate->parent_ = this;
      node_to_operate->depth_  = this->depth_ + 1;
      return node_to_operate.get();
    }

   public:
    explicit Node(ValueType t_pt, ValueComp const& t_cmp = {}) : pt_(std::move(t_pt)), pt_comp_(t_cmp) {}

    [[nodiscard]] auto value() const noexcept { return this->pt_; }

    [[nodiscard]] auto depth() const noexcept { return this->depth_; }

    [[nodiscard]] auto is_leaf() const noexcept {
      return this->child_node_[0] == nullptr and this->child_node_[1] == nullptr;
    }
  };

  KDTreeIterator<KDTree> begin() { return KDTreeIterator<KDTree>(this, this->get_left_most()); }

  KDTreeIterator<KDTree> end() { return KDTreeIterator<KDTree>(this); }

  void clear() {
    this->root_.reset();
    this->leaf_count_ = 0;
    this->size_       = 0;
  }

  void insert(ValueType const& t_pt) noexcept {
    if (root_ == nullptr) {
      this->root_       = std::make_unique<Node>(t_pt, this->pt_cmp_);
      this->size_       = 1;
      this->leaf_count_ = 1;
      return;
    }

    if (auto* const result = this->root_->insert(this, t_pt); result != nullptr) {
      ++this->size_;
    }
  }

  [[nodiscard]] std::size_t get_leaf_count() const noexcept { return this->leaf_count_; }

  [[nodiscard]] bool empty() const noexcept { return this->size_ == 0; }
  [[nodiscard]] std::size_t size() const noexcept { return this->size_; }

  // TODO should be taking universal reference
  [[nodiscard]] Node* find_node(ValueType const& t_pt) const noexcept {
    auto* traverse = this->root_.get();
    while (traverse != nullptr) {
      if (traverse->pt_ == t_pt) {
        break;
      }

      traverse = pt_cmp_(traverse->pt_, t_pt, &traverse->pivot_) ? traverse->child_node_[0].get()
                                                                 : traverse->child_node_[1].get();
    }

    return traverse;
  }

  explicit KDTree(ValueComp t_cmp = {}) : pt_cmp_(t_cmp) {}

 private:
  Node* get_left_most() const noexcept {
    Node* ret_val = this->root_.get();
    while (ret_val != nullptr and not ret_val->is_leaf()) {
      ret_val = ret_val->child_node_[0].get();
    }

    return ret_val;
  }

  std::size_t size_       = 0;
  std::size_t leaf_count_ = 0;
  ValueComp pt_cmp_;

  std::unique_ptr<Node> root_{};
};

}  // namespace epf

#endif