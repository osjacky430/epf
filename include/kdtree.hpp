#ifndef KDTREE_HPP_
#define KDTREE_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace particle_filter {

/**
 *  @todo 1. add tree iterator
 */
template <std::size_t Dimension, typename Point, typename ValueComp>
class KDTree {
  using ValueType = Point;

 public:
  class Node {
    friend class KDTree;

    std::array<std::unique_ptr<Node>, 2> child_node_;

    ValueType pt_;
    ValueComp pt_comp_;

    std::size_t depth_ = 0;

    Node* insert(KDTree* const t_base, ValueType const& t_pt) noexcept {
      if (this->is_leaf() and this->pt_ == t_pt) {
        this->pt_.update(t_pt);
        return nullptr;
      }

      auto& node_to_operate = pt_comp_(this->pt_, t_pt) ? this->child_node_[0] : this->child_node_[1];
      if (node_to_operate != nullptr) {
        return node_to_operate->insert(t_base, t_pt);
      }

      t_base->leaf_count_ += static_cast<std::size_t>(not this->is_leaf());
      node_to_operate         = std::make_unique<Node>(t_pt, this->pt_comp_);
      node_to_operate->depth_ = this->depth_ + 1;
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

  [[nodiscard]] Node* find_node(ValueType const& t_pt) const noexcept {
    auto* traverse = this->root_.get();
    while (traverse != nullptr) {
      if (traverse->pt_ == t_pt) {
        break;
      }

      traverse = pt_cmp_(traverse->pt_, t_pt) ? traverse->child_node_[0].get() : traverse->child_node_[1].get();
    }

    return traverse;
  }

  explicit KDTree(ValueComp t_cmp = {}) : pt_cmp_(t_cmp) {}

 private:
  std::size_t size_       = 0;
  std::size_t leaf_count_ = 0;
  ValueComp pt_cmp_;

  std::unique_ptr<Node> root_{};
};

}  // namespace particle_filter

#endif