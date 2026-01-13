#ifndef LIVOX_DESKEW_CUDA__CUDA_MATH_CUH_
#define LIVOX_DESKEW_CUDA__CUDA_MATH_CUH_

#include <cuda_runtime.h>
#include <math.h>
#include "cuda_types.h"

// Quaternion operations for GPU
// Convention: q = (w, x, y, z) where w is scalar part

/**
 * @brief Quaternion multiplication: c = a * b
 */
__device__ __forceinline__ void quat_mul(
    float aw, float ax, float ay, float az,
    float bw, float bx, float by, float bz,
    float* cw, float* cx, float* cy, float* cz)
{
    *cw = aw * bw - ax * bx - ay * by - az * bz;
    *cx = aw * bx + ax * bw + ay * bz - az * by;
    *cy = aw * by - ax * bz + ay * bw + az * bx;
    *cz = aw * bz + ax * by - ay * bx + az * bw;
}

/**
 * @brief Quaternion conjugate (inverse for unit quaternion)
 */
__device__ __forceinline__ void quat_conjugate(
    float qw, float qx, float qy, float qz,
    float* qw_out, float* qx_out, float* qy_out, float* qz_out)
{
    *qw_out = qw;
    *qx_out = -qx;
    *qy_out = -qy;
    *qz_out = -qz;
}

/**
 * @brief Normalize quaternion
 */
__device__ __forceinline__ void quat_normalize(
    float* qw, float* qx, float* qy, float* qz)
{
    float norm = sqrtf(*qw * *qw + *qx * *qx + *qy * *qy + *qz * *qz);
    if (norm > 1e-8f) {
        float inv_norm = 1.0f / norm;
        *qw *= inv_norm;
        *qx *= inv_norm;
        *qy *= inv_norm;
        *qz *= inv_norm;
    }
}

/**
 * @brief Spherical linear interpolation (SLERP) between quaternions
 *
 * @param qw0, qx0, qy0, qz0 Start quaternion
 * @param qw1, qx1, qy1, qz1 End quaternion
 * @param t Interpolation factor [0, 1]
 * @param qw_out, qx_out, qy_out, qz_out Result quaternion
 */
__device__ __forceinline__ void quat_slerp(
    float qw0, float qx0, float qy0, float qz0,
    float qw1, float qx1, float qy1, float qz1,
    float t,
    float* qw_out, float* qx_out, float* qy_out, float* qz_out)
{
    // Compute dot product
    float dot = qw0 * qw1 + qx0 * qx1 + qy0 * qy1 + qz0 * qz1;

    // If dot < 0, negate one quaternion to take shorter path
    if (dot < 0.0f) {
        qw1 = -qw1;
        qx1 = -qx1;
        qy1 = -qy1;
        qz1 = -qz1;
        dot = -dot;
    }

    // If quaternions are very close, use linear interpolation
    const float DOT_THRESHOLD = 0.9995f;
    if (dot > DOT_THRESHOLD) {
        *qw_out = qw0 + t * (qw1 - qw0);
        *qx_out = qx0 + t * (qx1 - qx0);
        *qy_out = qy0 + t * (qy1 - qy0);
        *qz_out = qz0 + t * (qz1 - qz0);
        quat_normalize(qw_out, qx_out, qy_out, qz_out);
        return;
    }

    // Standard SLERP
    float theta_0 = acosf(dot);
    float theta = theta_0 * t;
    float sin_theta = sinf(theta);
    float sin_theta_0 = sinf(theta_0);

    float s0 = cosf(theta) - dot * sin_theta / sin_theta_0;
    float s1 = sin_theta / sin_theta_0;

    *qw_out = s0 * qw0 + s1 * qw1;
    *qx_out = s0 * qx0 + s1 * qx1;
    *qy_out = s0 * qy0 + s1 * qy1;
    *qz_out = s0 * qz0 + s1 * qz1;
}

/**
 * @brief Rotate point by quaternion: p' = q * p * q^-1
 *
 * Optimized version without explicit quaternion multiplication
 */
__device__ __forceinline__ void quat_rotate_point(
    float qw, float qx, float qy, float qz,
    float px, float py, float pz,
    float* px_out, float* py_out, float* pz_out)
{
    // t = 2 * cross(q.xyz, p)
    float tx = 2.0f * (qy * pz - qz * py);
    float ty = 2.0f * (qz * px - qx * pz);
    float tz = 2.0f * (qx * py - qy * px);

    // p' = p + q.w * t + cross(q.xyz, t)
    *px_out = px + qw * tx + (qy * tz - qz * ty);
    *py_out = py + qw * ty + (qz * tx - qx * tz);
    *pz_out = pz + qw * tz + (qx * ty - qy * tx);
}

/**
 * @brief Transform point by pose: p' = R*p + t
 */
__device__ __forceinline__ void transform_point(
    const Pose3D_GPU* pose,
    float px, float py, float pz,
    float* px_out, float* py_out, float* pz_out)
{
    // Rotate
    float rx, ry, rz;
    quat_rotate_point(pose->qw, pose->qx, pose->qy, pose->qz,
                      px, py, pz, &rx, &ry, &rz);

    // Translate
    *px_out = rx + pose->tx;
    *py_out = ry + pose->ty;
    *pz_out = rz + pose->tz;
}

/**
 * @brief Compute pose inverse: T_out = T_in^-1
 *
 * For pose T = (R, t), inverse is (R^T, -R^T * t)
 */
__device__ __forceinline__ void pose_inverse(
    const Pose3D_GPU* T_in,
    Pose3D_GPU* T_out)
{
    // Rotation inverse = conjugate (for unit quaternion)
    T_out->qw = T_in->qw;
    T_out->qx = -T_in->qx;
    T_out->qy = -T_in->qy;
    T_out->qz = -T_in->qz;

    // Translation: -R^T * t
    float rx, ry, rz;
    quat_rotate_point(T_out->qw, T_out->qx, T_out->qy, T_out->qz,
                      T_in->tx, T_in->ty, T_in->tz,
                      &rx, &ry, &rz);
    T_out->tx = -rx;
    T_out->ty = -ry;
    T_out->tz = -rz;
}

/**
 * @brief Compose two poses: T_out = T_a * T_b
 *
 * For poses T_a = (R_a, t_a) and T_b = (R_b, t_b):
 * T_out = (R_a * R_b, R_a * t_b + t_a)
 */
__device__ __forceinline__ void pose_compose(
    const Pose3D_GPU* T_a,
    const Pose3D_GPU* T_b,
    Pose3D_GPU* T_out)
{
    // Rotation: q_out = q_a * q_b
    quat_mul(T_a->qw, T_a->qx, T_a->qy, T_a->qz,
             T_b->qw, T_b->qx, T_b->qy, T_b->qz,
             &T_out->qw, &T_out->qx, &T_out->qy, &T_out->qz);

    // Translation: t_out = R_a * t_b + t_a
    float rt_x, rt_y, rt_z;
    quat_rotate_point(T_a->qw, T_a->qx, T_a->qy, T_a->qz,
                      T_b->tx, T_b->ty, T_b->tz,
                      &rt_x, &rt_y, &rt_z);
    T_out->tx = rt_x + T_a->tx;
    T_out->ty = rt_y + T_a->ty;
    T_out->tz = rt_z + T_a->tz;
}

/**
 * @brief Interpolate between two poses
 *
 * Translation: linear interpolation
 * Rotation: SLERP
 */
__device__ __forceinline__ void pose_interpolate(
    const Pose3D_GPU* p0,
    const Pose3D_GPU* p1,
    float alpha,
    Pose3D_GPU* p_out)
{
    // Linear interpolation for translation
    p_out->tx = (1.0f - alpha) * p0->tx + alpha * p1->tx;
    p_out->ty = (1.0f - alpha) * p0->ty + alpha * p1->ty;
    p_out->tz = (1.0f - alpha) * p0->tz + alpha * p1->tz;

    // SLERP for rotation
    quat_slerp(p0->qw, p0->qx, p0->qy, p0->qz,
               p1->qw, p1->qx, p1->qy, p1->qz,
               alpha,
               &p_out->qw, &p_out->qx, &p_out->qy, &p_out->qz);
}

#endif  // LIVOX_DESKEW_CUDA__CUDA_MATH_CUH_
