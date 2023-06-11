#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <xf86drm.h>
#include <xf86drmMode.h>

#include <jpeglib.h>
#include <jerror.h>

int load_jpeg(char* filename, void **data, size_t *size)
{
  unsigned long x, y;
  unsigned long data_size;
  unsigned char * rowptr[1];
  unsigned char * jdata;
  struct jpeg_decompress_struct info;
  struct jpeg_error_mgr err;

  FILE* file = fopen(filename, "rb");

  info.err = jpeg_std_error(& err);
  jpeg_create_decompress(& info);

  if(!file) {
     fprintf(stderr, "error reading JPEG file %s!", filename);
     return 1;
  }

  jpeg_stdio_src(&info, file);
  jpeg_read_header(&info, TRUE);

  jpeg_start_decompress(&info);

  x = info.output_width;
  y = info.output_height;
  //channels = info.num_components;

  data_size = x * y * 3;

  jdata = (unsigned char *)malloc(data_size);
  while (info.output_scanline < info.output_height) {
    rowptr[0] = (unsigned char *)jdata +
            3 * info.output_width * info.output_scanline; 

    jpeg_read_scanlines(&info, rowptr, 1);
  }

  jpeg_finish_decompress(&info);

  jpeg_destroy_decompress(&info);
  fclose(file);

  *data = jdata;
  *size = data_size;

  return 0;
}

struct modeset_dev {
        struct modeset_dev *next;

        uint32_t width;
        uint32_t height;
        uint32_t stride;
        uint32_t size;
        uint32_t handle;
        uint8_t *map;

        drmModeModeInfo mode;
        uint32_t fb;
        uint32_t conn;
        uint32_t crtc;
        drmModeCrtc *saved_crtc;
};

int main(int argc, char **argv)
{
        int fd_card;
        uint64_t has_dumb;
        drmModeRes *res;
        drmModeConnector *conn;
        drmModeEncoder *enc;
        struct modeset_dev dev;

        void *jpeg_data = 0;
        size_t jpeg_size = 0;

        if (argc != 2)
                return 1;

        if (load_jpeg(argv[1], &jpeg_data, &jpeg_size)) {
                fprintf(stderr, "failed to read jpeg\n");
                return 1;
        }

        printf("Loaded jpeg @ %p, size: %zu\n", jpeg_data, jpeg_size);

        const char *path_card = "/dev/dri/card0";

        fd_card = open(path_card, O_RDWR | O_CLOEXEC);

        if (drmGetCap(fd_card, DRM_CAP_DUMB_BUFFER, &has_dumb) < 0 || !has_dumb) {
                fprintf(stderr, "no dumb buffer support\n");
                return -1;
        }

        res = drmModeGetResources(fd_card);
        if (!res) {
                fprintf(stderr, "Failed to get resources\n");
                return -1;
        }

        for (int i = 0; i < res->count_connectors; ++i) {
                conn = drmModeGetConnector(fd_card, res->connectors[i]);
                if (!conn) {
                        fprintf(stderr, "failed to get connector %d\n", i);
                        continue;
                }

                memset(&dev, 0, sizeof(dev));
                dev.conn = conn->connector_id;

                if (conn->connection != DRM_MODE_CONNECTED) {
                        fprintf(stderr, "ignoring not connected connector\n");
                        // TODO: free conn
                        continue;
                }

                if (conn->count_modes == 0) {
                        fprintf(stderr, "no valid mode for connector %u", conn->connector_id);
                        // TODO free conn
                        continue;
                }

                memcpy(&dev.mode, &conn->modes[0], sizeof(dev.mode));
                dev.width = conn->modes[0].hdisplay;
                dev.height = conn->modes[0].vdisplay;
                fprintf(stderr, "mode for connector %u is %ux%u\n",
                        conn->connector_id, dev.width, dev.height);

                if (!conn->encoder_id)
                        fprintf(stderr, "null encoder for %u\n", conn->connector_id);

                enc = drmModeGetEncoder(fd_card, conn->encoder_id);

                if (!enc->crtc_id) {
                        fprintf(stderr, "no encodeer\n");
                        // TODO free conn
                        continue;
                }

                dev.crtc = enc->crtc_id;
                drmModeFreeEncoder(enc);

                struct drm_mode_create_dumb creq;
                struct drm_mode_map_dumb mreq;

                memset(&creq, 0, sizeof(creq));
                creq.width = dev.width;
                creq.height = dev.height;
                creq.bpp = 32;
                if (drmIoctl(fd_card, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0) {
                        fprintf(stderr, "failed to create dumb buffer\n");
                        // TODO free conn
                        return -1;
                }
                dev.stride = creq.pitch;
                dev.size = creq.size;
                dev.handle = creq.handle;

                printf("Allocated dumb buffer: %u %u %u\n", dev.stride, dev.size, dev.handle);

                if (drmModeAddFB(fd_card, dev.width, dev.height, 24, 32, dev.stride, dev.handle, &dev.fb)) {
                        fprintf(stderr, "failed to create framebuffer\n");
                        // TODO free conn
                        return -1;
                }

                memset(&mreq, 0, sizeof(mreq));
                mreq.handle = dev.handle;
                if (drmIoctl(fd_card, DRM_IOCTL_MODE_MAP_DUMB, &mreq)) {
                        fprintf(stderr, "failed to request map of dumb buffer\n");
                        // TODO free conn
                        return -1;
                }

                dev.map = mmap(0, dev.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_card, mreq.offset);
                if (dev.map == MAP_FAILED) {
                        fprintf(stderr, "failed to mmap");
                        return -1;
                }

                memset(dev.map, 0, dev.size);

                dev.saved_crtc = drmModeGetCrtc(fd_card, dev.crtc);
                if (drmModeSetCrtc(fd_card, dev.crtc, dev.fb, 0, 0, &dev.conn, 1, &dev.mode))
                        fprintf(stderr, "cannot set CRTC\n");

                break;
        }

        uint8_t * const pixels_begin = (uint8_t*) jpeg_data;
        uint8_t * const pixels_end = pixels_begin + jpeg_size;
        uint8_t * pixels_cur = pixels_begin;
        for (unsigned y = 0; y < dev.height; ++y) {
                for (unsigned x = 0; x < dev.width; ++x) {
                        const uint32_t pixel = (pixels_cur[0] << 16)
                                             | (pixels_cur[1] << 8 )
                                             | (pixels_cur[2] << 0 );
                        pixels_cur += 3;
                        if (pixels_cur >= pixels_end)
                                pixels_cur = pixels_begin;

                        *(uint32_t*)&dev.map[dev.stride * y + x * 4] = pixel;
                }
        }

        sleep(10);

        drmModeSetCrtc(
                fd_card, dev.saved_crtc->crtc_id, dev.saved_crtc->buffer_id,
                dev.saved_crtc->x, dev.saved_crtc->y, &dev.conn, 1,
                &dev.saved_crtc->mode
        );

        return 0;
}
