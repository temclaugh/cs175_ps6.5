#ifndef SGUTILS_H
#define SGUTILS_H

#include <vector>

#include "scenegraph.h"

struct RbtNodesFiller : public SgNodeVisitor {
  typedef std::vector<std::tr1::RigTForm > rbt_list;

  rbt_list& rbts;

  RbtNodesFiller(rbt_list& rbts) : rbts_(rbts) {}

  virtual bool visit(SgTransformNode& node) {
    using namespace std;
    using namespace tr1;
    shared_ptr<SgRbtNode> rbtPtr = dynamic_pointer_cast<SgRbtNode>(node);
    if (rbtPtr) {
      node->setRbt(rbts_.back());
      rbts_.pop_back();
    }
    return true;
  }
};

inline void fillSgRbtNodes(std::tr1::shared_ptr<SgNode> root, std::vector<std::tr1::RigTForm >& rbts) {
  RbtNodesFiller filler(rbts);
  root->accept(filler);
}

#endif
