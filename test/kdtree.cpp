#include "kdtree.hpp"

#include <gtest/gtest.h>
#include <utility>
#include <vector>

struct Point {
  static constexpr auto Dim = 3;
  explicit Point(std::array<double, Dim> const& t_p, double const t_weight = 0) : pt_{t_p}, weight_{t_weight} {}

  std::size_t pivot_idx_ = Dim + 1;
  int pivot_value_       = 0;

  std::array<double, Dim> pt_{};
  double weight_ = 0.0;

  friend Point operator-(Point const& t_lhs, Point const& t_rhs) noexcept {
    return Point{{
      t_lhs.pt_[0] - t_rhs.pt_[0],
      t_lhs.pt_[1] - t_rhs.pt_[1],
      t_lhs.pt_[2] - t_rhs.pt_[2],
    }};
  }

  friend bool operator==(Point const& t_lhs, Point const& t_rhs) noexcept { return t_lhs.pt_ == t_rhs.pt_; }

  void update(Point const& t_pt) noexcept { this->weight_ += t_pt.weight_; }
};

struct PointComp {
  bool operator()(Point& t_lhs, Point const& t_rhs) const noexcept {
    if (t_lhs.pivot_idx_ <= Point::Dim) {
      return t_lhs.pivot_value_ < t_rhs.pt_[t_lhs.pivot_idx_];
    }

    auto const& diff    = t_lhs - t_rhs;
    auto const max_diff = std::max(diff.pt_.begin(), diff.pt_.end(), [](double const* t_first, double const* t_sec) {
      return std::abs(*t_first) < std::abs(*t_sec);
    });
    t_lhs.pivot_idx_    = max_diff - diff.pt_.begin();
    t_lhs.pivot_value_  = (t_lhs.pt_[t_lhs.pivot_idx_] + t_rhs.pt_[t_lhs.pivot_idx_]) / 2.0;  // mean split

    return t_lhs.pivot_value_ < t_rhs.pt_[t_lhs.pivot_idx_];
  }
};

struct ExpectedValue {
  std::size_t depth;
  std::size_t size;
  std::size_t leaves;
};

void test_insertion(particle_filter::KDTree<Point, PointComp>& t_tree, Point const& t_pt, ExpectedValue t_v) {
  t_tree.insert(t_pt);
  auto const* const node = t_tree.find_node(t_pt);

  EXPECT_EQ(t_tree.size(), t_v.size);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->value(), t_pt);
  EXPECT_EQ(node->depth(), t_v.depth);
  EXPECT_EQ(t_tree.get_leaf_count(), t_v.leaves);
}

TEST(KDTree, insertion) {
  particle_filter::KDTree<Point, PointComp> tree;

  std::vector<std::pair<Point, ExpectedValue>> const test_sets{
    std::make_pair(Point{{1.0, 1.0, 0.0}}, ExpectedValue{0, 1, 1}),
    std::make_pair(Point{{3.0, 1.0, 0.0}}, ExpectedValue{1, 2, 1}),
    std::make_pair(Point{{1.0, 3.0, 0.0}}, ExpectedValue{1, 3, 2}),
    std::make_pair(Point{{5.0, 3.0, 0.0}}, ExpectedValue{2, 4, 2}),
  };

  for (auto const& test_set : test_sets) {
    test_insertion(tree, test_set.first, test_set.second);
  }
}

TEST(KDTree, update) {
  particle_filter::KDTree<Point, PointComp> tree;

  // TODO: RNG to generate random weight
  auto pt_to_insert = Point{{1.0, 1.0, 0.0}};
  tree.insert(pt_to_insert);

  {
    auto const* const node = tree.find_node(pt_to_insert);
    EXPECT_EQ(node->value().weight_, 0.0);
  }

  pt_to_insert.weight_ = 1.0;
  tree.insert(pt_to_insert);

  {
    auto const* const node = tree.find_node(pt_to_insert);
    EXPECT_EQ(node->value().weight_, 1.0);
  }

  // update for non-leaf node
  auto new_pt_to_insert = Point{{2.0, 1.0, 0.0}};
  tree.insert(new_pt_to_insert);

  pt_to_insert.weight_ = 1.0;
  tree.insert(pt_to_insert);

  {
    auto const* const node = tree.find_node(pt_to_insert);
    EXPECT_EQ(node->value().weight_, 2.0);
  }
}

TEST(KDTree, in_order_traverse) {
  particle_filter::KDTree<Point, PointComp> tree;
  std::for_each(tree.begin(), tree.end(), [](auto const& /**/) { /* do nothing */ });
}