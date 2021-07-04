//
// Created by yiak on 2021/7/1.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_POINT3D_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_POINT3D_H

#include <memory>

#include <unordered_map>
using std::unordered_map;
using std::pair;

#include <base/io/sensors/velodyne_points.h>
#include <base/misc.h>

#include "base/logging.h"

namespace svso {
namespace pose_graph {

using namespace svso::base::io::reader;

class Frame;
class WorldPoint3D;

// World point3d either extracted from lidar or estiamted from camera
class MapPoint {
public:
    using Ptr = std::shared_ptr<MapPoint>;
    using ConstPtr = std::shared_ptr<const MapPoint>;
    using WeakPtr = std::weak_ptr<MapPoint>;

    MapPoint() {
        Init();
    }

    explicit MapPoint(double x, double y, double z=0) : MapPoint() {
        update(x, y, z);

    }
    virtual ~MapPoint() {}

    void Init() {
        identity.reset(new base::Identity(MapPoint::seq() ) );
        p.reset(new Point3D());
        is_bad_ = false;
    }

    /*
     * attributes
     */
    base::Identity::Ptr identity;
    Point3D::Ptr p;

    static base::AtomicCounter seq;

    virtual void update(double x, double y, double z) {
        p->x = x;
        p->y = y;
        p->z = z;
    }

    virtual void set_bad_flag() {
        is_bad_ = true;
    }

    virtual bool is_bad() {
        return is_bad_;
    }

    virtual double x() const {
        return p->x;
    }

    virtual double y() const {
        return p->y;
    }

    virtual double z() const {
        return p->z;
    }

private:
    bool is_bad_;
};

class CameraPoint3D: public MapPoint {
public:
    using Type = MapPoint;
    using Ptr = std::shared_ptr<Type>;
    using WeakPtr = std::weak_ptr<Type>;

    CameraPoint3D() : MapPoint() {
        Init();
    }

    explicit CameraPoint3D(double x, double y, double z) : CameraPoint3D() {
        update(x, y, z);
    }

    void Init() {
        identity.reset(new base::Identity(CameraPoint3D::seq() ) );
        p.reset(new Point3D());
        is_triangulated = false;
    }

    /*
     * attributes
     */
    base::Identity::Ptr identity;
    Point3D::Ptr p;
    std::weak_ptr<Frame> frame;
    bool is_triangulated;

    // topology
    WeakPtr parent;
    std::weak_ptr<WorldPoint3D> world;

    static base::AtomicCounter seq;

    void set_world(std::shared_ptr<WorldPoint3D> world_pt) {
        world = world_pt;
    }

    friend std::ostream& operator<< (std::ostream& out, const CameraPoint3D& v) {
        out << format("<CameraPoint %d: %.3f, %.3f %.3f>", v.identity->seq(),
                      v.x(),
                      v.y(),
                      v.z());
        return out;
    }

};

class WorldPoint3D : public MapPoint {
public:
    WorldPoint3D() : MapPoint() {
        Init();
    }

    explicit WorldPoint3D(double x, double y, double z) : WorldPoint3D() {
        update(x, y, z);
    }

    void Init() {
        identity.reset(new base::Identity(WorldPoint3D::seq() ) );
        p.reset(new Point3D());
    }

    /*
     * attributes
     */
    base::Identity::Ptr identity;
    Point3D::Ptr p;

    // topology
    using FrameKey = int64_t;
    using CamPt = int32_t;
    using FramesObservation = std::unordered_map<FrameKey, CamPt>;
    FramesObservation observations;

    static base::AtomicCounter seq;

    void associate_with(std::shared_ptr<Frame> frame, CameraPoint3D::Ptr cam_pt3d)
    {
        NOT_IMPLEMENTED
    }

    friend std::ostream& operator<< (std::ostream& out, const WorldPoint3D& v) {
        out << format("<WorldPoint3D %d: %.3f, %.3f %.3f>", v.identity->seq(),
                      v.x(),
                      v.y(),
                      v.z());
        return out;
    }
};

  } // pose_graph
} // svso

#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_POINT3D_H
