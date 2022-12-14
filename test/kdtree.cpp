#include "epf/core/kdtree.hpp"

#include <gtest/gtest.h>
#include <utility>
#include <vector>

struct Point {
  static constexpr auto Dim = 3;
  explicit Point(std::array<double, Dim> const& t_p, double const t_weight = 0) : pt_{t_p}, weight_{t_weight} {}

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

template <>
struct std::equal_to<Point> {
  bool operator()(Point& t_lhs, Point const& t_rhs) const noexcept {
    bool ret_val = (t_lhs == t_rhs);
    if (ret_val) {
      t_lhs.update(t_rhs);
    }

    return ret_val;
  }
};

struct PointComp {
  bool operator()(Point const& t_lhs, Point const& t_rhs,
                  std::pair<double, std::size_t>* const t_pivot) const noexcept {
    if (t_pivot->second < Point::Dim) {
      return t_pivot->first < t_rhs.pt_[t_pivot->second];
    }

    auto const& diff = t_lhs - t_rhs;
    auto const* const max_diff =
      std::max(diff.pt_.begin(), diff.pt_.end(),
               [](double const* t_first, double const* t_sec) { return std::abs(*t_first) < std::abs(*t_sec); });
    t_pivot->second = max_diff - diff.pt_.begin();
    t_pivot->first  = (t_lhs.pt_[t_pivot->second] + t_rhs.pt_[t_pivot->second]) / 2.0;  // mean split

    return t_pivot->first < t_rhs.pt_[t_pivot->second];
  }
};

struct ExpectedValue {
  std::size_t depth;
  std::size_t size;
  std::size_t leaves;
};

void test_insertion(epf::KDTree<Point, PointComp>& t_tree, Point const& t_pt, ExpectedValue t_v) {
  t_tree.insert(t_pt);
  auto const* const node = t_tree.find_node(t_pt);

  EXPECT_EQ(t_tree.size(), t_v.size);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->value(), t_pt);
  EXPECT_EQ(node->depth(), t_v.depth);
  EXPECT_EQ(t_tree.get_leaf_count(), t_v.leaves);
}

TEST(KDTree, insertion) {
  epf::KDTree<Point, PointComp> tree;

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
  epf::KDTree<Point, PointComp> tree;

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
  epf::KDTree<Point, PointComp> tree;
  std::for_each(tree.begin(), tree.end(), [](auto const& /**/) { /* do nothing */ });
}